import os
import sys
import argparse

# ─────────────────── Config ───────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="cpu")
parser.add_argument("--model_type", type=str, default="8")
parser.add_argument("--model_family", type=str, default="llama3base")
parser.add_argument("--strategy", type=str, default="multi_layer",
                    choices=["original", "multi_layer"],
                    help="Feature extraction strategy")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer, chat_format_modern, find_answer_start, get_pe
from utils.multi_layer import extract_features, get_feature_keys
import json
from tqdm import tqdm

model_type = args.model_type
model_family = args.model_family
strategy = args.strategy
model_name = f"{model_family}{model_type}b"

print(f"Model: {model_name} | Strategy: {strategy}")


def prompt_chat_for_labeled(prompt):
    return [{"role": "user", "content": prompt}]


def get_tokenized_ids(answer, q):
    """Tokenize prompt + answer and find where the answer starts."""

    # LLaMA 3 chat
    if model_family == "llama3chat":
        prompt_ids = chat_format_modern(prompt_chat_for_labeled(q), tokenizer)
        full_ids = chat_format_modern(prompt_chat_for_labeled(q), tokenizer, answer.strip())
        start_at = find_answer_start(prompt_ids, full_ids)
        return full_ids, start_at

    # Old LLaMA 2 chat
    if "chat" in model_family:
        full_ids = chat_change_with_answer(
            prompt_chat_for_labeled(q), answer.strip(), tokenizer
        )
        start_at = -1
        id1 = full_ids[0] if isinstance(full_ids[0], list) else full_ids
        for i in range(len(id1)):
            if id1[i:i+4] == [518, 29914, 25580, 29962]:
                start_at = i
        if start_at == -1:
            raise Exception("Could not find [/INST] tokens in llamachat")
        else:
            start_at += 4
        return full_ids, start_at

    # Base models (LLaMA 3 base, LLaMA 2 base, OPT, etc.)
    text = f"{q.strip()} {answer.strip()}"
    otext = q.strip()
    id1 = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
    id2 = tokenizer(otext.strip(), return_tensors='pt')['input_ids'].tolist()[0]

    start_at = -1
    for i in range(len(id1)):
        if i >= len(id2) or id1[i] != id2[i]:
            start_at = i
            break

    return [id1], start_at


def get_hd(answer, q):
    """Extract hidden state features + probability/entropy for a given answer."""
    ids, start_at = get_tokenized_ids(answer, q)
    op = model(torch.tensor(ids).to(model.device), output_hidden_states=True)
    hd = op.hidden_states

    # Extract features using selected strategy
    features = extract_features(hd, strategy=strategy, start_at=start_at)

    # Also compute probability and entropy (always useful)
    logit = op.logits
    flat_ids = ids[0] if isinstance(ids[0], list) else ids
    pl, el = get_pe(logit, flat_ids, start_at)

    return features, pl, el


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)

# ──────────────────────────────────────────────
# Process HELM data
# ──────────────────────────────────────────────
hd_result_path = f"./helm/hd/{model_name}"

if not os.path.exists(hd_result_path):
    os.makedirs(hd_result_path, exist_ok=True)

data_path = f"./helm/data/{model_name}/data.json"
if not os.path.exists(data_path):
    print(f"HELM data not found at {data_path}")
    print(f"Available model dirs: {os.listdir('./helm/data/') if os.path.exists('./helm/data/') else 'N/A'}")
    sys.exit(1)

with open(data_path, "r", encoding='utf-8') as f:
    data = json.load(f)

feature_keys = get_feature_keys(strategy)

result = {}
for d in tqdm(data, desc="Processing HELM data"):
    result[d] = {
        "sentences": [],
        "passage": None,
    }
    prompt = data[d]["prompt"]
    prompt = tokenizer.decode(
        tokenizer(prompt.strip(), return_tensors='pt')['input_ids'].tolist()[0],
        skip_special_tokens=True
    )

    for s in data[d]["sentences"]:
        features, pl, el = get_hd(s["sentence"], prompt)
        sentence_entry = {
            "probability": pl,
            "entropy": el,
        }
        # Add all feature keys
        for key in feature_keys:
            sentence_entry[key] = features[key]
        result[d]["sentences"].append(sentence_entry)

    # Passage-level features
    full_passage = " ".join([t["sentence"] for t in data[d]["sentences"]])
    features, pl, el = get_hd(full_passage, prompt)
    passage_entry = {
        "probability": pl,
        "entropy": el,
    }
    for key in feature_keys:
        passage_entry[key] = features[key]
    result[d]["passage"] = passage_entry

# Save with strategy suffix for clarity
out_file = f"{hd_result_path}/hd_{strategy}.json"
with open(out_file, "w+") as f:
    json.dump(result, f)
print(f"Saved: {out_file}")
