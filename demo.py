"""
Demo: generate a response to a paragraph and detect if it's hallucinated.

Usage:
    python demo.py --paragraph "Who is the president of united states?" \
                   --model_name llama3base8b \
                   --strategy multi_layer_mean \
                   --gpu 0

    # Run built-in true/false examples to verify the classifier:
    python demo.py --demo \
                   --model_name llama3base8b \
                   --strategy multi_layer_mean \
                   --gpu 0
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--paragraph",  type=str, default=None, help="Input paragraph/prompt")
parser.add_argument("--demo",       action="store_true", default=True,
                    help="Run built-in hallucinated and non-hallucinated examples")
parser.add_argument("--model_name", type=str, default="llama3base8b",
                    choices=["llama3base8b", "gptj"])
parser.add_argument("--strategy",   type=str, default="multi_layer",
                    choices=["original", "multi_layer", "multi_layer_last_token",
                             "multi_layer_mean", "multi_layer_deltas"])
parser.add_argument("--ckpt_path",  type=str, default=None,
                    help="Path to best_acc_model.pt. Defaults to the standard output path.")
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--gpu",        type=str, default="0")
parser.add_argument("--debug",      action="store_true", help="Print feature norms and start_at for diagnosis")
args = parser.parse_args()

if not args.demo and args.paragraph is None:
    parser.error("--paragraph is required unless --demo is set")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn.functional as F

from src.utils.model import get_model
from src.utils.multi_layer import get_model_config, get_input_size, get_feature_keys, extract_features
from src.utils.gen import find_answer_start, chat_format_modern
from src.classifier import HalluClassifier

# Built-in demo paragraphs: (paragraph, max_new_tokens)
DEMO_EXAMPLES = [
    ("Who is the preseident of united states?", 50),
    ("Is sun inside the earth?", 50),
    ("Who is vaibhav sankaran", 50),
]


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def infer_model_family_type(model_name):
    """Split e.g. 'llama3base8b' into ('llama3base', '8')."""
    for family in ["llama3base", "llama3chat", "llamabase", "llamachat", "falcon", "opt", "gptj"]:
        if model_name.startswith(family):
            suffix = model_name[len(family):]
            size = suffix.replace("b", "").replace("B", "")
            return family, size
    raise ValueError(f"Cannot infer model family from model_name: {model_name}")


def load_classifier(input_size, ckpt_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clf = HalluClassifier(input_size)
    state_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    clf.load_state_dict(state_dict)
    clf.to(device)
    clf.eval()
    return clf, device


def generate_response(model, tokenizer, generation_config, paragraph, model_family, max_new_tokens):
    """Greedy-decode a response to the paragraph."""
    config = {**generation_config, "max_new_tokens": max_new_tokens}

    if model_family.startswith("llama3") and "chat" in model_family:
        messages = [{"role": "user", "content": paragraph.strip()}]
        input_ids = chat_format_modern(messages, tokenizer)
    else:
        input_ids = tokenizer(paragraph.strip(), return_tensors="pt")["input_ids"].tolist()

    input_tensor = torch.tensor(input_ids).to(model.device)
    with torch.no_grad():
        out = model.generate(input_tensor, **config)

    # Decode only the newly generated tokens
    prompt_len = input_tensor.shape[1]
    new_ids = out.sequences[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def extract_hidden_features(model, tokenizer, paragraph, response, model_family, strategy):
    """Re-encode prompt+response and extract hidden-state features."""
    if model_family.startswith("llama3") and "chat" in model_family:
        messages = [{"role": "user", "content": paragraph.strip()}]
        prompt_ids = chat_format_modern(messages, tokenizer)
        full_ids = chat_format_modern(messages, tokenizer, response.strip())
        start_at = find_answer_start(prompt_ids, full_ids)
    else:
        prompt_text = paragraph.strip()
        full_text = f"{prompt_text} {response.strip()}"
        prompt_ids = [tokenizer(prompt_text, return_tensors="pt")["input_ids"].tolist()[0]]
        full_ids = [tokenizer(full_text, return_tensors="pt")["input_ids"].tolist()[0]]
        start_at = find_answer_start(prompt_ids, full_ids)

    if args.debug:
        flat_prompt = prompt_ids[0] if isinstance(prompt_ids[0], list) else prompt_ids
        flat_full   = full_ids[0]   if isinstance(full_ids[0],   list) else full_ids
        print(f"[debug] prompt tokens : {len(flat_prompt)}")
        print(f"[debug] full tokens   : {len(flat_full)}")
        print(f"[debug] start_at      : {start_at}")

    input_tensor = torch.tensor(full_ids).to(model.device)
    with torch.no_grad():
        op = model(input_tensor, output_hidden_states=True)

    features = extract_features(op.hidden_states, strategy=strategy, start_at=start_at)

    if args.debug:
        for k, v in features.items():
            t = torch.tensor(v)
            print(f"[debug] {k}: norm={t.norm():.4f}, mean={t.mean():.6f}")

    return features


def score(clf, device, features, feature_keys):
    feature_vec = [v for key in feature_keys for v in features[key]]
    input_ = torch.tensor([feature_vec]).to(device)
    with torch.no_grad():
        logits = clf(input_)
    if args.debug:
        print(f"[debug] raw logits: {logits.tolist()}")
    prob_hallu = F.softmax(logits, dim=1)[:, 1][0].item()
    return prob_hallu


def run_single(model, tokenizer, generation_config, clf, device, feature_keys,
               model_family, paragraph, max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = args.max_new_tokens
    print(f"\n--- Generating response ---")
    response = generate_response(model, tokenizer, generation_config, paragraph, model_family, max_new_tokens)
    print(f"Paragraph : {paragraph}")
    print(f"Response  : {response}")

    print("\n--- Extracting hidden-state features ---")
    features = extract_hidden_features(model, tokenizer, paragraph, response, model_family, args.strategy)

    print("--- Predicted Result ---")
    prob_hallu = score(clf, device, features, feature_keys)
    label = "HALLUCINATED" if prob_hallu >= 0.5 else "NOT HALLUCINATED"
    print(f"Prediction                : {label}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    model_family, model_type = infer_model_family_type(args.model_name)

    ckpt_path = args.ckpt_path or (
        f"./data/auto-labeled/output/{args.model_name}/{args.strategy}/train_log/best_acc_model.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Attempting to download from Google Drive...")
        fetch_script = os.path.join(os.path.dirname(__file__), "scripts", "fetch_checkpoints.sh")
        ret = os.system(f"bash {fetch_script} {args.model_name} {args.strategy}")
        if ret != 0 or not os.path.exists(ckpt_path):
            print("Download failed. Provide --ckpt_path or fill in scripts/fetch_checkpoints.sh.")
            sys.exit(1)

    config = get_model_config(args.model_name)
    input_size = get_input_size(config["hidden_dim"], config["num_layers"], args.strategy)
    feature_keys = get_feature_keys(args.strategy)

    print(f"Model       : {args.model_name}")
    print(f"Strategy    : {args.strategy}")
    print(f"Input size  : {input_size}")
    print(f"Checkpoint  : {ckpt_path}")
    print()

    print("Loading language model...")
    model, tokenizer, generation_config, _ = get_model(model_type, model_family, max_new_tokens=args.max_new_tokens)

    print("Loading hallucination classifier...")
    clf, device = load_classifier(input_size, ckpt_path)

    if args.demo:
        print("\n========== DEMO MODE ==========")
        for i, (paragraph, max_new_tokens) in enumerate(DEMO_EXAMPLES, 1):
            print(f"\n{'='*40}")
            print(f"Example {i}/{len(DEMO_EXAMPLES)}")
            run_single(model, tokenizer, generation_config, clf, device, feature_keys,
                       model_family, paragraph, max_new_tokens=max_new_tokens)
        print(f"\n{'='*40}")
    else:
        run_single(model, tokenizer, generation_config, clf, device, feature_keys,
                   model_family, args.paragraph)


if __name__ == "__main__":
    main()
