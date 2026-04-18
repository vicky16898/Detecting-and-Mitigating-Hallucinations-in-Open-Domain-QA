"""
Part B: Label HELM continuations for hallucination using gpt-4.1-mini via the
OpenAI Batch API (50% discount).

Three modes:
    --mode submit   : build batch input file, upload, create batch, save batch id
    --mode status   : poll batch status
    --mode fetch    : download results and merge labels into data.json

The judge sees (prompt, generated_sentence) and uses its own parametric knowledge
— no external Wikipedia fetch. Label: 1 = hallucination, 0 = faithful.

Usage:
    export OPENAI_API_KEY=...
    python label_helm_data.py --model_name llama3base8b --mode submit
    python label_helm_data.py --model_name llama3base8b --mode status
    python label_helm_data.py --model_name llama3base8b --mode fetch
"""

import os
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True,
                    help="e.g. llama3base8b — reads/writes ./helm/data/{model_name}/data.json")
parser.add_argument("--mode", choices=["submit", "status", "fetch"], required=True)
parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
args = parser.parse_args()

from openai import OpenAI

client = OpenAI()

data_dir = Path(f"./helm/data/{args.model_name}")
data_path = data_dir / "data.json"
batch_input_path = data_dir / "batch_input.jsonl"
batch_output_path = data_dir / "batch_output.jsonl"
batch_id_path = data_dir / "batch_id.txt"

SYSTEM_PROMPT = (
    "You are a fact-checking judge. You will be given a Wikipedia-style context "
    "prompt and a single sentence that a language model generated as a continuation. "
    "Decide whether the generated sentence contains a hallucination — i.e. a factual "
    "claim that is incorrect, fabricated, or unsupported by real-world knowledge "
    "about the subject of the prompt. "
    "If the sentence contains any hallucinated factual claim, label=1. "
    "If the sentence is faithful, trivially true, or contains no verifiable factual "
    "claims (e.g. generic filler), label=0. "
    "Respond with strict JSON only: {\"label\": 0 or 1}."
)


def build_user_message(prompt, sentence):
    return (
        f"Context prompt:\n{prompt}\n\n"
        f"Generated sentence:\n{sentence}\n\n"
        f"Output JSON with the label field only."
    )


def custom_id(wiki_id, sent_idx):
    return f"{wiki_id}__{sent_idx}"


def parse_custom_id(cid):
    wiki_id, sent_idx = cid.rsplit("__", 1)
    return wiki_id, int(sent_idx)


# ──────────────────────────────────────────────
def mode_submit():
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = 0
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for wiki_id, entry in data.items():
            prompt = entry["prompt"]
            for i, s in enumerate(entry["sentences"]):
                if s.get("label") is not None:
                    continue
                body = {
                    "model": args.judge_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_message(prompt, s["sentence"])},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 20,
                    "temperature": 0,
                }
                req = {
                    "custom_id": custom_id(wiki_id, i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(req) + "\n")
                n += 1
    print(f"Wrote {n} requests to {batch_input_path}")

    uploaded = client.files.create(file=open(batch_input_path, "rb"), purpose="batch")
    print(f"Uploaded file id: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"model_name": args.model_name},
    )
    with open(batch_id_path, "w") as f:
        f.write(batch.id)
    print(f"Batch created: {batch.id} (status: {batch.status})")
    print(f"Saved batch id to {batch_id_path}")


def mode_status():
    batch_id = open(batch_id_path).read().strip()
    batch = client.batches.retrieve(batch_id)
    print(f"Batch {batch_id}")
    print(f"  status: {batch.status}")
    print(f"  request_counts: {batch.request_counts}")
    if batch.output_file_id:
        print(f"  output_file_id: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"  error_file_id: {batch.error_file_id}")


def mode_fetch():
    batch_id = open(batch_id_path).read().strip()
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch not completed yet (status={batch.status}). Re-run later.")
        return
    content = client.files.content(batch.output_file_id).text
    with open(batch_output_path, "w", encoding="utf-8") as f:
        f.write(content)

    labels = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec["custom_id"]
        try:
            msg = rec["response"]["body"]["choices"][0]["message"]["content"]
            parsed = json.loads(msg)
            label = int(parsed["label"])
            if label not in (0, 1):
                label = 1
        except Exception as e:
            print(f"Parse failure for {cid}: {e}; defaulting to 1")
            label = 1
        labels[cid] = label

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filled = 0
    for wiki_id, entry in data.items():
        for i, s in enumerate(entry["sentences"]):
            cid = custom_id(wiki_id, i)
            if cid in labels and s.get("label") is None:
                s["label"] = labels[cid]
                filled += 1

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Filled {filled} labels into {data_path}")


if args.mode == "submit":
    mode_submit()
elif args.mode == "status":
    mode_status()
elif args.mode == "fetch":
    mode_fetch()
