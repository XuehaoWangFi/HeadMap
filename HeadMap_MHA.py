import argparse
import sys
import random
import numpy as np

import torch
import transformers
from datasets import Dataset

sys.path.append('lm-evaluation-harness/')
import lm_eval.tasks
import lm_eval.models
from transformers import LlamaForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--knowledge_circuit_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--head_num', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--scaling', default=1.0, type=float, required=True)
    return parser.parse_args()

args = parse_args()

base_model = args.base_model

model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map='cuda',
    )
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

seed=args.seed
data_name=args.data_name

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

knowledge_circuit = torch.load(args.knowledge_circuit_dir)['knowledge_circuit']
output_dir = args.out_dir

from transformers.models.llama.modeling_llama import *
from types import MethodType
import torch.nn as nn

r = args.rank
train_on_inputs = True
add_eos_token = False
cutoff_len = 512
micro_batch_size = 16
gradient_accumulation_steps = 1
num_epochs = 2
learning_rate = args.lr
val_set_size = 2000
ddp = False
group_by_length = False


for layer_idx in knowledge_circuit:
    retain_heads = knowledge_circuit[layer_idx]['retain_heads']
    head_dim = model.model.layers[int(layer_idx)].self_attn.head_dim
    hidden_size = model.model.layers[int(layer_idx)].self_attn.hidden_size
    model.model.layers[int(layer_idx)].self_attn.retain_heads = retain_heads
    model.model.layers[int(layer_idx)].self_attn.retain_heads_map = nn.Sequential(nn.Linear(len(retain_heads) * head_dim, r, bias=True), nn.Linear(r, hidden_size, bias=True))
    nn.init.kaiming_uniform_(model.model.layers[int(layer_idx)].self_attn.retain_heads_map[0].weight)
    nn.init.zeros_(model.model.layers[int(layer_idx)].self_attn.retain_heads_map[1].weight)
    nn.init.zeros_(model.model.layers[int(layer_idx)].self_attn.retain_heads_map[1].bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
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
            retain_heads_output = attn_output[:, :, self.retain_heads, :]
            retain_heads_output = retain_heads_output.view(bsz, q_len, -1)
            residual_output = self.retain_heads_map(retain_heads_output)

            attn_output = attn_output.view(bsz, q_len, -1)
            
            attn_output = self.o_proj(attn_output) + residual_output * args.scaling

            return attn_output, None, past_key_value
    
    model.model.layers[int(layer_idx)].self_attn.forward = MethodType(forward, model.model.layers[int(layer_idx)].self_attn)

for n, p in model.named_parameters():
    p.requires_grad = False
    if 'retain_heads_map' in n:
        p.requires_grad = True

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point['init_text']
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = data_point['input']
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

task_dict = lm_eval.tasks.get_task_dict([data_name])
task = task_dict[data_name]

train_task_docs = list(task.training_docs())
train_datalist = []
for doc_id, doc in enumerate(train_task_docs):
    train_datalist.append({
        'input': task.doc_to_text(doc),
        'init_text': task.doc_to_text(doc) + task.doc_to_target(doc),
        'init_label': task.doc_to_target(doc)
    })
train_data = Dataset.from_list(train_datalist).shuffle().map(generate_and_tokenize_prompt)


if val_set_size > 0:
    if task.has_validation_docs() and task.has_test_docs():
        print('use validation set')
        val_task_doc = list(task.validation_docs())
        val_datalist = []
        for doc_id, doc in enumerate(val_task_doc):
            val_datalist.append({
                'input': task.doc_to_text(doc),
                'init_text': task.doc_to_text(doc) + task.doc_to_target(doc),
                'init_label': task.doc_to_target(doc)
            })
        train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
        val_data = Dataset.from_list(val_datalist).shuffle().map(generate_and_tokenize_prompt)
    else:
        print('use 10% train set as validation set')
        val_set_size = int(0.1 * len(train_data))
        train_val = train_data.train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
else:
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
    val_data = None

if train_data:
    train_data = train_data.remove_columns(['input', 'init_text', 'init_label'])

if val_data:
    val_data = val_data.remove_columns(['input', 'init_text', 'init_label'])
    
print(train_data, val_data)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=200,
        save_total_limit=1,
        output_dir=output_dir,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

def get_state_dict(model, state_dict=None):
    to_return = {k: state_dict[k] for k in state_dict if "retain_heads_map" in k}
    return to_return

model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

trainer.train(resume_from_checkpoint=False)

lm = lm_eval.models.get_model('hf-causal-experimental').create_from_arg_string(f'pretrained={base_model}', {"batch_size": 32, "device": 'cuda'})
lm.model = model
lm.model.eval()

from lm_eval.evaluator import *

task_dict = lm_eval.tasks.get_task_dict([data_name])

results = evaluate(
    lm=lm,
    task_dict=task_dict,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
)
print(results)