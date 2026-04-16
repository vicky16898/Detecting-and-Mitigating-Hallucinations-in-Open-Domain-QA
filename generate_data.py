import os
import argparse

# ─────────────────── Config ───────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--model_type", type=str, default="8")
parser.add_argument("--model_family", type=str, default="llama3base")
parser.add_argument("--topk_first_token", type=int, default=4)
parser.add_argument("--windows", type=int, default=16)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_type = args.model_type
model_family = args.model_family

wiki_path = "./auto-labeled/wiki"
output_path = f"./auto-labeled/output/{model_family}{model_type}b"

topk_first_token = args.topk_first_token
topk_next_token = topk_first_token
windows = args.windows
# ──────────────────────────────────────────────

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer, chat_format_modern
from tqdm import tqdm
import json
import spacy

model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# Subword prefix character varies by tokenizer
if "llama" in model_family or "baichuan" in model_family:
    st = "▁"
else:
    st = "Ġ"

nlp = spacy.load('en_core_web_sm')
prompt_chat = []


def delete_substrings(lst):
    substrings = []
    lst = list(set(lst))
    for s in lst:
        if any(s in o for o in lst if o != s):
            substrings.append(s)
    for s in substrings:
        lst.remove(s)
    return lst


def find_boundaries(text, words):
    boundaries = []
    for word in words:
        start = 0
        ntext = text
        while True:
            start = ntext.find(word)
            if start == -1:
                break
            end = start + len(word) - 1
            while start > 0 and ntext[start - 1] != " ":
                start -= 1
            while end < len(ntext) - 1 and ntext[end + 1] != " ":
                end += 1
            boundaries.append("".join([ntext[i] for i in range(start, end + 1)]))
            ntext = ntext[end + 1:]
    return boundaries


def get_entities(text):
    entities_ = list(set([str(e) for e in nlp(text).ents]))
    entities_ = find_boundaries(text, entities_)
    entities = delete_substrings(entities_)
    all_entities = []
    for i in range(len(text)):
        for e in entities:
            if text[i:].startswith(e):
                all_entities.append((e, i))
    return all_entities


def get_prompt_prefix(title):
    """Get the prompt prefix text for the given model family."""
    if model_family == "vicuna":
        return (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
            f"USER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: "
        )
    return ""


def get_chat_messages(title):
    """Get chat messages for chat model families."""
    return [
        {"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer: "}
    ]


def find_first_and_next_token(text, e, idx, input_id, prompt=""):
    """Find the first token of the entity and the token after @ marker."""
    new_text = f"{text[:idx].strip()} {text[idx:].replace(e, e + ' @', 1).strip()}"

    # Tokenize with @ marker inserted
    if model_family.startswith("llama3") and "chat" not in model_family:
        new_input_id = tokenizer(prompt + new_text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
    elif model_family == "falcon":
        new_input_id = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        # Falcon-specific handling
        correct_id = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        new_with_at = tokenizer(prompt + new_text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        for i in range(len(input_id[0])):
            if input_id[0][i] != new_with_at[i]:
                return []
        ap = 0
        for i in range(len(new_with_at)):
            if i >= len(correct_id):
                return []
            if correct_id[i] != new_with_at[i]:
                next_token = correct_id[i]
                ap = i
                break
        first_token = new_with_at[len(input_id[0])]
        try:
            return [first_token, next_token, ap - 1 - len(input_id[0]), correct_id[ap:]]
        except:
            return []
    else:
        new_input_id = tokenizer(prompt + new_text.strip(), return_tensors='pt')['input_ids'].tolist()[0]

    # Check that prefix tokens match
    for i in range(len(input_id[0])):
        if input_id[0][i] != new_input_id[i]:
            return []

    first_token = new_input_id[len(input_id[0])] # Token immediately after the prefix (first token of the entity)

    if isinstance(at_id, list):
        at_position = None
        for i in range(len(input_id[0]), len(new_input_id)):  # start after prompt
            if new_input_id[i] in at_id:
                at_position = i
                break
        if at_position is None:
            print(f"None of the @ token IDs {at_id} were found in the new input IDs.")
            return []
    else:
        try:
            print(f"Looking for @ token ID {at_id} in new input IDs...")
            at_position = new_input_id.index(at_id)
        except ValueError:
            # @ token not found — might be tokenized differently in LLaMA 3
            # Try searching for all possible @ token IDs
            at_candidates = tokenizer.encode("@", add_special_tokens=False)
            at_position = len(new_input_id) - 1
            for candidate in at_candidates:
                if candidate in new_input_id[len(input_id[0]):]:
                    at_position = new_input_id.index(candidate, len(input_id[0]))
                    break

    if at_position == len(new_input_id) - 1:
        return []
    next_token = new_input_id[at_position + 1]
    return [first_token, next_token, at_position - len(input_id[0]), new_input_id[at_position + 1:]]


def find_first_and_next_token_for_chat(text, e, idx, input_id, title):
    """Find tokens for chat models (LLaMA 3 chat, LLaMA 2 chat)."""
    new_text = f"{text[:idx].strip()} {text[idx:].replace(e, e + ' @', 1).strip()}"

    messages = get_chat_messages(title)

    if model_family.startswith("llama3"):
        new_input_id = chat_format_modern(messages, tokenizer, new_text.strip())[0]
    else:
        new_input_id = chat_change_with_answer(messages, new_text.strip(), tokenizer)[0]

    for i in range(len(input_id[0])):
        if input_id[0][i] != new_input_id[i]:
            return []

    first_token = new_input_id[len(input_id[0])]

    # Find @ position
    at_candidates = tokenizer.encode("@", add_special_tokens=False)
    at_position = len(new_input_id) - 1
    for candidate in at_candidates:
        if candidate in new_input_id[len(input_id[0]):]:
            at_position = new_input_id.index(candidate, len(input_id[0]))
            break

    if at_position == len(new_input_id) - 1:
        return []
    next_token = new_input_id[at_position + 1]
    return [first_token, next_token, at_position - len(input_id[0]), new_input_id[at_position + 1:]]


# ──────────────────────────────────────────────
# Main data generation loop
# ──────────────────────────────────────────────
print(f"Model: {model_family}{model_type}b (device = {model.device})")
print(f"Output: {output_path}")

for data_type in ["train", "valid", "test"]:
    result = []
    wiki_file = f"{wiki_path}/wiki_{data_type}.json"
    if not os.path.exists(wiki_file):
        print(f"Skipping {data_type}: {wiki_file} not found")
        continue

    with open(wiki_file, encoding='utf-8') as f:
        data = json.load(f)

    for ii, d in tqdm(enumerate(data), total=len(data), desc=f"Generating {data_type}"):
        text = " ".join(d["sentences"][:2])
        entities_ = get_entities(text)

        entities = []
        idx_ = []
        for e in entities_:
            if e[1] not in idx_:
                idx_.append(e[1])
                entities.append(e)

        mytexts = []
        new_entities = []
        original_entity = []
        ret = {
            "original_text": text,
            "title": d["title"],
        }
        chat_messages = get_chat_messages(d["title"])

        for e, idx in entities:
            if idx == 0 or e in d["title"]:
                continue

            # Tokenize the prefix up to the entity
            is_chat = "chat" in model_family
            if not is_chat:
                p_ = get_prompt_prefix(d["title"])
                input_id = tokenizer(
                    p_ + text[:idx].strip(), return_tensors='pt'
                )['input_ids'].tolist()
                tokens = find_first_and_next_token(text, e, idx, input_id, p_)
            else:
                if model_family.startswith("llama3"):
                    input_id = chat_format_modern(
                        chat_messages, tokenizer, text[:idx].strip()
                    )
                else:
                    input_id = chat_change_with_answer(
                        chat_messages, text[:idx].strip(), tokenizer
                    )
                tokens = find_first_and_next_token_for_chat(
                    text, e, idx, input_id, d["title"]
                )

            if not tokens:
                continue
            first_, next_, entity_len, last_id = tokens

            # Generate continuation from the prefix
            output = model.generate(
                torch.tensor(input_id).to(model.device), **generation_config
            )
            values, indices = torch.topk(output.scores[0], k=topk_first_token)
            if first_ in indices[0].tolist():
                continue
            sequences = output.sequences

            for i in range(entity_len + windows):
                output = model.generate(sequences, **generation_config)
                values, indices = torch.topk(output.scores[0], k=topk_next_token)
                if next_ in indices[0].tolist():
                    break
                sequences = output.sequences

            if next_ not in indices[0].tolist():
                continue

            new_sequence = sequences[0].tolist()
            new_entity_id = new_sequence[len(input_id[0]):]

            # Build the new text with hallucinated entity marked by @
            if model_family == "falcon":
                all_new_text_id = (
                    input_id[0] + [204, 43, 204] + new_entity_id + [204, 43, 204] + last_id
                )
            elif isinstance(at_id, list):
                all_new_text_id = (
                    input_id[0] + [at_id[0]] + new_entity_id + [at_id[0]] + last_id
                )
            else:
                all_new_text_id = (
                    input_id[0] + [at_id] + new_entity_id + [at_id] + last_id
                )

            mytext = tokenizer.decode(all_new_text_id, skip_special_tokens=True)
            new_entity = mytext[mytext.find("@") + 1: mytext.rfind("@")].strip().lower()

            if any(ee.strip() in text.lower() for ee in new_entity.split(" ")) or e.lower() in new_entity:
                continue

            # Extract just the answer part for chat models
            if model_family == "vicuna":
                mytext = mytext.split("ASSISTANT:")[-1].strip()
            if model_family == "llamachat":
                mytext = mytext.split("[/INST]")[-1].strip()
            if model_family == "llama3chat":
                # For LLaMA 3 chat, extract content after the last assistant header
                if "assistant\n\n" in mytext:
                    mytext = mytext.split("assistant\n\n")[-1].strip()
                elif "assistant" in mytext.lower():
                    parts = mytext.split("assistant")
                    mytext = parts[-1].strip()

            mytexts.append(mytext)
            new_entities.append(new_entity)
            original_entity.append((e, idx))

        ret["texts"] = mytexts
        ret["new_entities"] = new_entities
        ret["original_entities"] = original_entity
        result.append(ret)

    out_file = f"{output_path}/data_{data_type}.json"
    with open(out_file, "w+", encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    print(f"Saved: {out_file} ({len(result)} samples)")