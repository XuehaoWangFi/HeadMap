import argparse
import fnmatch
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('lm-evaluation-harness/')
import lm_eval.tasks as tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    return parser.parse_args()

args = parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map='cuda',
)
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token_id = (
    0
)
tokenizer.padding_side = "left"

train_on_inputs = False
def tokenize(prompt, add_eos_token=False):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 512
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
            user_prompt, add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if False:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

task_names = pattern_match([args.data_name], tasks.ALL_TASKS)
task_dict = tasks.get_task_dict(task_names)
train_datalist = []
import random
for task_name in task_names:
    task = task_dict[task_name]
    train_task_docs = list(task.training_docs())
    rnd = random.Random()
    rnd.seed(42)
    rnd.shuffle(train_task_docs)
    for doc_id, doc in enumerate(train_task_docs):
        doc_text = task.doc_to_text(doc)
        train_datalist.append({
            'input': doc_text,
            'init_text': doc_text + task.doc_to_target(doc),
            'init_label': task.doc_to_target(doc)
        })
model.eval()
with torch.no_grad():
    losses = []
    for sample in train_datalist:
        tokenize_sample = generate_and_tokenize_prompt(sample)
        loss = model(torch.Tensor(tokenize_sample['input_ids']).long().reshape(1, -1).cuda(), torch.Tensor(tokenize_sample['attention_mask']).long().reshape(1, -1).cuda(), labels=torch.Tensor(tokenize_sample['labels']).long().reshape(1, -1).cuda()).loss
        losses.append(loss.item())
torch.save({'losses': losses, 'samples': train_datalist}, args.out_dir)