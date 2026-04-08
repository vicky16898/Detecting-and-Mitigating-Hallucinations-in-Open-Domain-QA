"""
Multi-layer hidden state feature extraction utilities.

Strategies:
  - "original":    Reproduces the original MIND paper behavior
                   (avg all layers for last token + last layer mean across tokens)
                   → 2 * hidden_size dimensions

  - "multi_layer": Concatenate hidden states from selected representative layers
                   (last token from layers: 1, N//4, N//2, 3N//4, N)
                   + mean across tokens from first, mid, last layers
                   + layer deltas (late - early)
                   → much richer feature set
"""

import torch


# ─────────────────────────────────────────────────────────
# Layer Selection
# ─────────────────────────────────────────────────────────

def select_layer_indices(num_layers, strategy="multi_layer"):
    """
    Select which layer indices to extract hidden states from.

    Args:
        num_layers: Total number of transformer layers (e.g., 32 for LLaMA)
        strategy: "original" or "multi_layer"

    Returns:
        dict with keys describing which layers to use for each feature type

    Note: hidden_states[0] is the embedding layer, hidden_states[1] is layer 1, etc.
          So for a 32-layer model, hidden_states indices go from 0 to 32.
    """
    if strategy == "original":
        return {
            "last_token_layers": list(range(1, num_layers + 1)),  # all layers (1..N)
            "mean_token_layers": [num_layers],                     # last layer only
        }

    elif strategy == "multi_layer":
        # Pick 5 representative layers: first, 1/4, 1/2, 3/4, last
        quarter = max(1, num_layers // 4)
        half = max(1, num_layers // 2)
        three_quarter = max(1, (3 * num_layers) // 4)

        return {
            # For last-token features: 5 specific layers concatenated
            "last_token_layers": [1, quarter, half, three_quarter, num_layers],

            # For mean-across-tokens features: first, mid, last layers
            "mean_token_layers": [1, half, num_layers],

            # For delta features: (late - upper_mid) and (lower_mid - early)
            "delta_pairs": [
                (num_layers, three_quarter),   # late delta
                (half, 1),                      # early delta
            ],
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ─────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────

def extract_features(hidden_states, strategy="multi_layer", start_at=0):
    """
    Extract feature vectors from hidden states.

    Args:
        hidden_states: Tuple of tensors from model output.
                       hidden_states[i] has shape [batch, seq_len, hidden_dim].
                       hidden_states[0] = embedding layer, [1..N] = transformer layers.
        strategy: "original" or "multi_layer"
        start_at: Token position where the answer/generated text begins.
                  Used for mean-pooling across answer tokens only.

    Returns:
        dict of feature name → list (suitable for JSON serialization)
    """
    num_layers = len(hidden_states) - 1  # subtract embedding layer
    layer_info = select_layer_indices(num_layers, strategy)

    features = {}

    if strategy == "original":
        # ── Original Feature 1: last token, averaged across ALL layers ──
        hds = hidden_states[1][0][-1].clone().detach()
        for i in range(2, num_layers + 1):
            hds += hidden_states[i][0][-1].clone().detach()
        hds = hds / num_layers
        features["hd_last_token"] = hds.tolist()

        # ── Original Feature 2: last layer, mean across answer tokens ──
        hds_mean = torch.mean(hidden_states[num_layers][0][max(0, start_at - 1):], dim=0)
        features["hd_last_mean"] = hds_mean.tolist()

    elif strategy == "multi_layer":
        # ── Feature 1: Last token from each selected layer (CONCATENATED) ──
        last_token_parts = []
        for layer_idx in layer_info["last_token_layers"]:
            vec = hidden_states[layer_idx][0][-1].clone().detach()
            last_token_parts.append(vec)
        features["hd_multi_last_token"] = torch.cat(last_token_parts).tolist()

        # ── Feature 2: Mean across tokens from first, mid, last layers ──
        mean_parts = []
        for layer_idx in layer_info["mean_token_layers"]:
            vec = torch.mean(
                hidden_states[layer_idx][0][max(0, start_at - 1):], dim=0
            )
            mean_parts.append(vec)
        features["hd_multi_mean"] = torch.cat(mean_parts).tolist()

        # ── Feature 3: Layer deltas (difference between layers) ──
        delta_parts = []
        for (high_layer, low_layer) in layer_info["delta_pairs"]:
            high_vec = hidden_states[high_layer][0][-1].clone().detach()
            low_vec = hidden_states[low_layer][0][-1].clone().detach()
            delta_parts.append(high_vec - low_vec)
        features["hd_deltas"] = torch.cat(delta_parts).tolist()

    return features


# ─────────────────────────────────────────────────────────
# Input Size Computation
# ─────────────────────────────────────────────────────────

def get_input_size(hidden_dim, num_layers, strategy="multi_layer"):
    """
    Compute the total input size for the MLP classifier.

    Args:
        hidden_dim: Hidden dimension of the model (e.g., 4096 for LLaMA)
        num_layers: Number of transformer layers (e.g., 32 for LLaMA)
        strategy: "original" or "multi_layer"

    Returns:
        int: Total input feature dimension
    """
    if strategy == "original":
        # hd_last_token (hidden_dim) + hd_last_mean (hidden_dim)
        return hidden_dim * 2

    elif strategy == "multi_layer":
        layer_info = select_layer_indices(num_layers, strategy)

        # hd_multi_last_token: one hidden_dim per selected layer
        last_token_dim = len(layer_info["last_token_layers"]) * hidden_dim

        # hd_multi_mean: one hidden_dim per selected layer
        mean_dim = len(layer_info["mean_token_layers"]) * hidden_dim

        # hd_deltas: one hidden_dim per delta pair
        delta_dim = len(layer_info["delta_pairs"]) * hidden_dim

        return last_token_dim + mean_dim + delta_dim

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ─────────────────────────────────────────────────────────
# Model Config Lookup
# ─────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    # Old models (for backward compatibility)
    "llamabase7b":   {"hidden_dim": 4096, "num_layers": 32},
    "llamachat7b":   {"hidden_dim": 4096, "num_layers": 32},
    "llamabase13b":  {"hidden_dim": 5120, "num_layers": 40},
    "llamachat13b":  {"hidden_dim": 5120, "num_layers": 40},
    "opt7b":         {"hidden_dim": 4096, "num_layers": 32},
    "falcon7b":      {"hidden_dim": 4544, "num_layers": 32},
    "gptj":          {"hidden_dim": 4096, "num_layers": 28},

    # New models
    "llama3base8b":  {"hidden_dim": 4096, "num_layers": 32},
    "llama3chat8b":  {"hidden_dim": 4096, "num_layers": 32},
}


def get_model_config(model_name):
    """Look up hidden_dim and num_layers for a given model name."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Fallback: try to infer from name
    if "7b" in model_name or "8b" in model_name:
        hidden_dim = 4544 if "falcon" in model_name else 4096
        num_layers = 32
    elif "13b" in model_name:
        hidden_dim = 5120
        num_layers = 40
    else:
        hidden_dim = 4096
        num_layers = 32

    return {"hidden_dim": hidden_dim, "num_layers": num_layers}


def get_feature_keys(strategy):
    """Return the ordered list of feature keys for a given strategy."""
    if strategy == "original":
        return ["hd_last_token", "hd_last_mean"]
    elif strategy == "multi_layer":
        return ["hd_multi_last_token", "hd_multi_mean", "hd_deltas"]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
