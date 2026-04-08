from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch


def get_model(model_type, model_family, max_new_tokens=1):
    """
    Load a model and tokenizer.

    Args:
        model_type: Model size identifier (e.g., "7b", "8b", "8")
        model_family: Model family name. Supported:
            NEW:  "llama3base", "llama3chat"
            OLD:  "llamabase", "llamachat", "opt", "falcon", "gptj", "mpt",
                  "vicuna", "bloom", "baichuan"
        max_new_tokens: Max tokens to generate (for generation_config)

    Returns:
        model, tokenizer, generation_config, at_id
    """
    at_id = None

    # ──────────────────────────────────────────────
    # Determine model path
    # ──────────────────────────────────────────────

    # === NEW MODELS ===
    if model_family == "llama3base":
        model_path = f"meta-llama/Llama-3.1-{model_type}B"

    elif model_family == "llama3chat":
        model_path = f"meta-llama/Llama-3.1-{model_type}B-Instruct"

    # === OLD MODELS (backward compatible) ===
    elif model_family == "opt":
        model_path = "facebook/opt-6.7b" if model_type == "7b" else "facebook/opt-13b"
        at_id = [787, 1039]
    elif model_family == "llamabase":
        model_path = f"meta-llama/Llama-2-{model_type}-hf"
    elif model_family == "bloom":
        model_path = "bigscience/bloom-7b1"
        at_id = [2566, 35]
    elif model_family == "falcon":
        at_id = 0
        model_path = f"tiiuae/falcon-{model_type}"
    elif model_family == "gptj":
        model_path = "EleutherAI/gpt-j-6b"
        at_id = 2488
    elif model_family == "mpt":
        model_path = "mosaicml/mpt-7b"
        at_id = [1214, 33]
    elif model_family == "vicuna":
        model_path = f"lmsys/vicuna-{model_type}-v1.5"
        at_id = 732
    elif model_family == "llamachat":
        model_path = f"meta-llama/Llama-2-{model_type}-chat-hf"
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    # ──────────────────────────────────────────────
    # Load model and tokenizer (universal for new models)
    # ──────────────────────────────────────────────

    if model_family.startswith("llama3"):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Dynamically find @ token ID from tokenizer
        at_id = tokenizer.encode("@", add_special_tokens=False)
        if len(at_id) == 1:
            at_id = at_id[0]
        # else at_id stays as a list (multi-token)

        # Set pad token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Old LLaMA 2 loading
    elif "llama" in model_family:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        at_id = 732

    # Baichuan
    elif "baichuan" in model_family:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", trust_remote_code=True
        )
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        if tokenizer.pad_token_id == 64000:
            tokenizer.pad_token_id = 0
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.float32,
            trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto"
        )
        at_id = [3757, 92952]

    # MPT
    elif "mpt" in model_family:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # All other old models (OPT, Falcon, GPT-J, Bloom, etc.)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ──────────────────────────────────────────────
    # Generation config
    # ──────────────────────────────────────────────

    generation_config = dict(
        top_k=0,
        top_p=1.0,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=True,
    )

    # Some models need explicit eos/pad token IDs to avoid warnings
    if model_family in ("falcon", "gptj") or model_family.startswith("llama3"):
        generation_config["eos_token_id"] = tokenizer.eos_token_id
        generation_config["pad_token_id"] = tokenizer.pad_token_id or tokenizer.eos_token_id

    return model, tokenizer, generation_config, at_id