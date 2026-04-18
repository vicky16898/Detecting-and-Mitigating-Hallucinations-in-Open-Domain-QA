import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse

# ─────────────────── Config ───────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--model_type", type=str, default="8")
parser.add_argument("--model_family", type=str, default="llama3base")
parser.add_argument("--strategy", type=str, default="multi_layer",
                    choices=["original", "multi_layer", "multi_layer_last_token", "multi_layer_mean", "multi_layer_deltas"],
                    help="Feature extraction strategy")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_type = args.model_type
model_family = args.model_family
strategy = args.strategy
model_path = f"./data/auto-labeled/output/{model_family}{model_type}b"
result_path = f"{model_path}/{args.strategy}"
# ──────────────────────────────────────────────

print(f"Model: {model_family}{model_type}b | Strategy: {strategy}")

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer, chat_format_modern, find_answer_start
from utils.multi_layer import extract_features, get_feature_keys
from tqdm import tqdm
import json


model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)


def prompt_chat(title):
    return [{"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer: "}]


def get_tokenized_ids(otext, title=None):
    """Tokenize text, handling different model families."""
    text = otext.replace("@", "").replace("  ", " ").replace("  ", " ")
    text = tokenizer.decode(
        tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0],
        skip_special_tokens=True
    )

    if model_family == "vicuna":
        text = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            f"USER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: {text}"
        )
        return tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()

    # LLaMA 3 chat
    if model_family == "llama3chat":
        return chat_format_modern(prompt_chat(title), tokenizer, text.strip())

    # Old LLaMA 2 chat
    if "chat" in model_family:
        return chat_change_with_answer(prompt_chat(title), text.strip(), tokenizer)

    # Base models (LLaMA 3 base, LLaMA 2 base, etc.)
    return tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()


def get_answer_start(ids, title=None):
    """Find where the answer tokens begin in the token sequence."""
    # LLaMA 3 chat: use apply_chat_template to find boundary
    if model_family == "llama3chat":
        prompt_only_ids = chat_format_modern(prompt_chat(title), tokenizer)
        return find_answer_start(prompt_only_ids, ids)

    # Old LLaMA 2 chat: look for [/INST] tokens
    if model_family == "llamachat":
        for i in range(len(ids[0])):
            if ids[0][i:i+4] == [518, 29914, 25580, 29962]:
                return i + 4
        print("WARNING: [/INST] not found in llamachat tokens")
        return 1

    # Vicuna
    if model_family == "vicuna":
        prompt_text = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            f"USER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: "
        )
        prompt_ids = tokenizer(prompt_text)['input_ids']
        for i in range(len(ids[0])):
            if i >= len(prompt_ids) or ids[0][i] != prompt_ids[i]:
                return i
        return len(prompt_ids)

    # Base models: answer starts right from the beginning (skip BOS)
    return 2


def get_hd(text, title=None):
    """Extract hidden state features from a text."""
    ids = get_tokenized_ids(text, title)
    start_at = get_answer_start(ids, title)

    # Forward pass with hidden states
    output = model(
        torch.tensor(ids).to(model.device),
        output_hidden_states=True
    )
    hd = output.hidden_states

    # Extract features using selected strategy
    features = extract_features(hd, strategy=strategy, start_at=start_at)
    return features


# ──────────────────────────────────────────────
# Main: Process train/valid/test splits
# ──────────────────────────────────────────────

feature_keys = get_feature_keys(strategy)

os.makedirs(result_path, exist_ok=True)

for data_type in ["train", "valid", "test"]:
    data_path = f"{model_path}/data_{data_type}.json"
    if not os.path.exists(data_path):
        print(f"Skipping {data_type}: {data_path} not found")
        continue

    data = json.load(open(data_path, encoding='utf-8'))

    # Initialize result containers for each feature
    results = {key: [] for key in feature_keys}

    for k in tqdm(data, desc=f"Processing {data_type}"):
        sample_results = {key: {"right": None, "hallu": []} for key in feature_keys}

        # Extract features from the original (correct) text
        origin_features = get_hd(k["original_text"], k["title"])
        for key in feature_keys:
            sample_results[key]["right"] = origin_features[key]

        # Extract features from each hallucinated text
        for t in k["texts"]:
            hallu_features = get_hd(t, k["title"])
            for key in feature_keys:
                sample_results[key]["hallu"].append(hallu_features[key])

        for key in feature_keys:
            results[key].append(sample_results[key])

    # Save results
    for key in feature_keys:
        out_path = f"{result_path}/{key}_{data_type}.json"
        with open(out_path, "w+") as f:
            json.dump(results[key], f)
        print(f"Saved: {out_path}")
