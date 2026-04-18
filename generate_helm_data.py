"""
Part A: Generate HELM continuations for a new model.

Reads prompts from ./helm/prompt_mapping.txt, greedy-decodes 128 tokens per prompt,
sentence-splits the continuation, and writes ./helm/data/{model_name}/data.json
with label=None placeholders to be filled in by label_helm_data.py.

Usage:
    python generate_helm_data.py --model_family llama3base --model_type 8 --gpu 0
"""

import os
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--model_type", type=str, default="8")
parser.add_argument("--model_family", type=str, default="llama3base")
parser.add_argument("--prompt_file", type=str, default="./helm/prompt_mapping.txt")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import spacy
from tqdm import tqdm
from utils.model import get_model
from utils.gen import chat_format_modern

model_name = f"{args.model_family}{args.model_type}b"
out_dir = f"./helm/data/{model_name}"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/data.json"

# ──────────────────────────────────────────────
# Load prompt_mapping.txt — format: "{wiki_id}: {prompt}"
# ──────────────────────────────────────────────
prompts = {}
with open(args.prompt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        sep = line.find(":")
        if sep == -1:
            continue
        wiki_id = line[:sep].strip()
        prompt_text = line[sep + 1:].strip()
        prompts[wiki_id] = prompt_text

print(f"Loaded {len(prompts)} prompts")

# ──────────────────────────────────────────────
# Load model + sentence splitter
# ──────────────────────────────────────────────
model, tokenizer, _, _ = get_model(args.model_type, args.model_family, max_new_tokens=128)
model.eval()

nlp = spacy.load("en_core_web_sm")

generation_config = dict(
    top_k=0,
    top_p=1.0,
    do_sample=False,
    num_beams=1,
    max_new_tokens=128,
    return_dict_in_generate=True,
    pad_token_id=tokenizer.eos_token_id,
)


def build_input_ids(prompt):
    if args.model_family == "llama3chat":
        messages = [{"role": "user", "content": prompt.strip()}]
        return chat_format_modern(messages, tokenizer)
    if "chat" in args.model_family:
        from utils.gen import chat_change
        return chat_change([{"role": "user", "content": prompt.strip()}], tokenizer)
    return tokenizer(prompt.strip(), return_tensors="pt")["input_ids"].tolist()


def decode_continuation(full_ids, prompt_ids):
    """Return just the newly-generated text, stripping the prompt."""
    prompt_len = len(prompt_ids[0])
    new_ids = full_ids[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ──────────────────────────────────────────────
# Generate
# ──────────────────────────────────────────────
data = {}
with torch.no_grad():
    for wiki_id, prompt in tqdm(prompts.items(), desc="Generating"):
        input_ids = build_input_ids(prompt)
        ids_tensor = torch.tensor(input_ids).to(model.device)
        attn_mask = torch.ones_like(ids_tensor)
        out = model.generate(ids_tensor, attention_mask=attn_mask, **generation_config)
        full_ids = out.sequences.tolist()
        continuation = decode_continuation(full_ids, input_ids).strip()

        sentences = [s.text.strip() for s in nlp(continuation).sents if s.text.strip()]

        data[wiki_id] = {
            "prompt": prompt,
            "sentences": [{"sentence": s, "label": None} for s in sentences],
        }

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(data)} entries to {out_path}")
