import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Old-style chat formatting (LLaMA 2 / Vicuna)
# ──────────────────────────────────────────────

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]


def chat_change(dialog, tokenizer):
    """Format a dialog for LLaMA 2 chat (prompt only, no answer)."""
    prompt_tokens = []
    dialog_tokens = sum(
        [
            tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            ) + [2]
            for prompt, answer in zip(dialog[::2], dialog[1::2])
        ],
        [],
    )
    assert dialog[-1]["role"] == "user"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
    )
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def chat_change_with_answer(dialog, answer_, tokenizer):
    """Format a dialog for LLaMA 2 chat (prompt + answer appended)."""
    prompt_tokens = []
    dialog_tokens = sum(
        [
            tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            ) + [2]
            for prompt, answer in zip(dialog[::2], dialog[1::2])
        ],
        [],
    )
    assert dialog[-1]["role"] == "user"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST} {answer_.strip()}",
    )
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens


# ──────────────────────────────────────────────
# New-style chat formatting (LLaMA 3.1 and other modern models)
# Works universally via HuggingFace's apply_chat_template
# ──────────────────────────────────────────────

def chat_format_modern(messages, tokenizer, answer=None):
    """
    Format messages using HuggingFace's apply_chat_template.
    Works for LLaMA 3.1, Gemma 2, and any model with a chat template.

    Args:
        messages: List of {"role": ..., "content": ...} dicts
        tokenizer: HuggingFace tokenizer with chat template
        answer: Optional answer string to append as assistant response

    Returns:
        List of token IDs (as a list, not tensor)
    """
    if answer is not None:
        # Append the answer as an assistant message
        messages = messages + [{"role": "assistant", "content": answer}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors=None
        )
    else:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors=None
        )
    # Extract input_ids from Encoding object if needed
    if hasattr(ids, 'input_ids'):
        ids = ids.input_ids
    return [ids]  # wrap in list to match old format: [[id1, id2, ...]]


def find_answer_start(prompt_ids, full_ids):
    """
    Find where the answer begins by comparing prompt-only IDs vs full (prompt+answer) IDs.
    Universal method that works for any tokenizer.

    Args:
        prompt_ids: Token IDs of just the prompt (no answer)
        full_ids: Token IDs of prompt + answer

    Returns:
        int: Index where the answer tokens start
    """
    # Handle nested lists
    if isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    if isinstance(full_ids[0], list):
        full_ids = full_ids[0]

    for i in range(len(full_ids)):
        if i >= len(prompt_ids) or full_ids[i] != prompt_ids[i]:
            return i
    return len(prompt_ids)


# ──────────────────────────────────────────────
# Shared generation utility
# ──────────────────────────────────────────────

def generate_output(model_family, model, tokenizer, config, text, answer=None):
    """Generate model output for a given text/dialog."""
    if model_family.startswith("llama3"):
        # Modern models: use apply_chat_template for chat, plain text for base
        if "chat" in model_family:
            messages = [{"role": "user", "content": text.strip()}]
            if answer is None:
                input_id = chat_format_modern(messages, tokenizer)
            else:
                input_id = chat_format_modern(messages, tokenizer, answer.strip())
        else:
            input_id = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()
    elif "chat" in model_family:
        if answer is None:
            input_id = chat_change(
                [{"role": "user", "content": text.strip()}], tokenizer
            )
        else:
            input_id = chat_change_with_answer(
                [{"role": "user", "content": text.strip()}], answer.strip(), tokenizer
            )
    else:
        assert answer is None
        input_id = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()

    output = model.generate(torch.tensor(input_id).to(model.device), **config)
    return output


# ──────────────────────────────────────────────
# Probability and Entropy computation
# ──────────────────────────────────────────────

def get_pe(logit, id_, start_at):
    """Compute token probabilities and entropy."""
    probabilities = F.softmax(logit, dim=2)
    log_probabilities = torch.log(probabilities)
    entropy = -probabilities * log_probabilities
    entropy_sum = torch.sum(entropy, dim=-1)

    pl = []
    el = []
    for i, idx in enumerate(id_[1:]):
        if i < start_at - 1:
            continue
        pl.append(probabilities[0][i][idx].item())
        el.append(entropy_sum[0][i].item())
    return pl, el