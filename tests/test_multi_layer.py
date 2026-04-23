import pytest
import torch
import sys
import os

# Add src to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.multi_layer import (
    select_layer_indices,
    get_input_size,
    get_model_config,
    get_feature_keys,
    extract_features
)

def test_select_layer_indices_original():
    num_layers = 32
    indices = select_layer_indices(num_layers, strategy="original")
    assert indices["last_token_layers"] == list(range(1, 33))
    assert indices["mean_token_layers"] == [32]

def test_select_layer_indices_multi_layer():
    num_layers = 32
    indices = select_layer_indices(num_layers, strategy="multi_layer")
    assert indices["last_token_layers"] == [1, 8, 16, 24, 32]
    assert indices["mean_token_layers"] == [1, 16, 32]
    assert indices["delta_pairs"] == [(32, 24), (16, 1)]

def test_get_input_size():
    hidden_dim = 4096
    num_layers = 32
    
    # Original: 2 * hidden_dim
    assert get_input_size(hidden_dim, num_layers, "original") == 4096 * 2
    
    # multi_layer_last_token: 5 * hidden_dim
    assert get_input_size(hidden_dim, num_layers, "multi_layer_last_token") == 4096 * 5
    
    # multi_layer_mean: 3 * hidden_dim
    assert get_input_size(hidden_dim, num_layers, "multi_layer_mean") == 4096 * 3
    
    # multi_layer_deltas: 2 * hidden_dim
    assert get_input_size(hidden_dim, num_layers, "multi_layer_deltas") == 4096 * 2
    
    # multi_layer (all): 5 + 3 + 2 = 10 * hidden_dim
    assert get_input_size(hidden_dim, num_layers, "multi_layer") == 4096 * 10

def test_get_model_config():
    # Known model
    llama3 = get_model_config("llama3base8b")
    assert llama3["hidden_dim"] == 4096
    assert llama3["num_layers"] == 32
    
    # Fallback 7b
    fallback_7b = get_model_config("some_random_7b_model")
    assert fallback_7b["hidden_dim"] == 4096
    assert fallback_7b["num_layers"] == 32
    
    # Fallback 13b
    fallback_13b = get_model_config("other_13b_model")
    assert fallback_13b["hidden_dim"] == 5120
    assert fallback_13b["num_layers"] == 40

def test_get_feature_keys():
    assert get_feature_keys("original") == ["hd_last_token", "hd_last_mean"]
    assert get_feature_keys("multi_layer") == ["hd_multi_last_token", "hd_multi_mean", "hd_deltas"]
    with pytest.raises(ValueError):
        get_feature_keys("invalid_strategy")

def test_extract_features_shape():
    # Mock hidden states for a 2-layer model
    # hidden_states[0] is embedding, [1] is L1, [2] is L2
    hidden_dim = 16
    seq_len = 5
    batch_size = 1 # extract_features assumes [0] indexing for batch
    
    hidden_states = [
        torch.randn(batch_size, seq_len, hidden_dim) for _ in range(3)
    ]
    
    # Original strategy for 2-layer model
    features = extract_features(hidden_states, strategy="original", start_at=0)
    
    # last_token: average of L1 and L2 last tokens
    assert "hd_last_token" in features
    assert len(features["hd_last_token"]) == hidden_dim
    
    # last_mean: mean of L2 tokens
    assert "hd_last_mean" in features
    assert len(features["hd_last_mean"]) == hidden_dim

    # Multi-layer strategy
    # For num_layers=2:
    # last_token_layers: [1, 1, 1, 1, 2] -> wait, select_layer_indices logic:
    # quarter = 2//4 = 0 -> max(1, 0) = 1
    # half = 2//2 = 1
    # three_quarter = (3*2)//4 = 1
    # layers: [1, 1, 1, 1, 2]
    features_ml = extract_features(hidden_states, strategy="multi_layer", start_at=0)
    assert "hd_multi_last_token" in features_ml
    # 5 layers * hidden_dim
    assert len(features_ml["hd_multi_last_token"]) == 5 * hidden_dim
