import argparse
import fnmatch
import os
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from accelerate import Accelerator
 
sys.path.append('lm-evaluation-harness/')
import lm_eval.tasks as tasks
 
class MultiChoice:
    def __init__(self, choices):
        self.choices = choices
 
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False
 
        return True
 
    def __iter__(self):
        for choice in self.choices:
            yield choice
 
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--save_path", default="./results/")
    parser.add_argument("--tasks", required=True, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--sample_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument('--train_on_inputs', default=False, action='store_true') # set false
    parser.add_argument('--add_eos_token', default=False, action='store_true') # set false
 
    parser.add_argument("--chosen_layers", type=str, default="0-31")
    parser.add_argument("--head_num", type=int, default=8)
 
    return parser.parse_args()
 
 
from transformers.models.llama.modeling_llama import *
from types import MethodType
 
 
def evaluate(model, batchs):
    losses = []
    accelerator = Accelerator()
    for batch in batchs:
        outputs = model(**batch)
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.Tensor(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
 
 
def masking(
    lm,  # The model
    batchs,
    layers_w_heads=None, # [(layer_idx, [head_idx_1, head_idx_2]),...]
): 
    prev_forward = {}
    for layer_idx, corrupt_heads in layers_w_heads:
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            corrupt_heads: List[Tuple] = corrupt_heads,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            bsz, q_len, _ = hidden_states.size()
 
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
 
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
 
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
 
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
 
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
 
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
 
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
 
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False
 
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            # bsz * seq_len * head_num * head_dim
             
            bsz, seq_len, head_num, head_dim = attn_output.shape
            head_mask = torch.ones([bsz, seq_len, head_num, head_dim])
            for corrupt_head in corrupt_heads:
                head_mask[:, :, corrupt_head, :] = 0
            attn_output *= head_mask.to(attn_output.device)
 
            attn_output = attn_output.view(bsz, q_len, -1)
             
            attn_output = self.o_proj(attn_output)
 
            return attn_output, None, past_key_value
 
        prev_forward[layer_idx] = lm.model.layers[layer_idx].self_attn.forward
        lm.model.layers[layer_idx].self_attn.forward = MethodType(forward, lm.model.layers[layer_idx].self_attn)
     
    loss, perplexity = evaluate(
        lm, batchs
    )
 
    for layer_idx, corrupt_heads in layers_w_heads:
        lm.model.layers[layer_idx].self_attn.forward = prev_forward[layer_idx]
    return loss


def layer_conditioned_locating(model, batchs, head_num=8, chosen_layers=[0, 31]):
    table = []
    path = []
 
    num_module = len(model.model.layers)

    head_num_ = [head_num for _ in range(num_module)]
    for layer_idx in range(chosen_layers[0], chosen_layers[1] + 1):
        row = []
        for head_idx in range(model.model.layers[layer_idx].self_attn.num_heads):
            r = masking(
                model,
                batchs,
                layers_w_heads=path + [(layer_idx, [head_idx])],
            )
            row.append(r)
        table.append(torch.Tensor(row))

        tmp = table[-1]
        path += [(layer_idx, torch.topk(tmp, k=head_num_[layer_idx]).indices)]
        table[-1] = tmp
    return dict(
        scores=torch.stack(table).T.detach().cpu(), 
        path=path)
 
 
 
def main():
    args = parse_args()
    device_map = "auto"
     
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"
 
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
 
        result["labels"] = result["input_ids"].copy()
 
        return result
 
    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point['init_text']
        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = data_point['input']
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
 
            if args.add_eos_token:
                user_prompt_len -= 1
 
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
     
    task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    train_datalist = []

    info = torch.load(args.sample_dir)
    losses = info['losses']
    samples = info['samples']
    
    indices = torch.topk(-1 * torch.Tensor(losses), k=args.limit).indices
    for idx in indices:
        train_datalist.append(samples[idx])
    
    train_data = Dataset.from_list(train_datalist)
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
    train_data = train_data.remove_columns(['input', 'init_text', 'init_label'])
 
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=0,
            warmup_steps=100,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=200,
            output_dir="./",
            save_total_limit=20,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    eval_dataloader = trainer.get_train_dataloader()
    batchs = []
    for batch in eval_dataloader:
        batchs.append(batch)
 
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
     
    model.eval()
    with torch.no_grad():
        results = layer_conditioned_locating(
            model, batchs, head_num=args.head_num, chosen_layers=[int(_) for _ in args.chosen_layers.split('-')],
        )
 
    knowledge_circuit = {}
 
    for layer_idx, retain_head in results['path']:
        knowledge_circuit[layer_idx] = {
            'retain_heads': retain_head.tolist()
        }
     
    results["knowledge_circuit"] = knowledge_circuit
 
    torch.save(results, args.save_path + "/info.pt")
 
main()