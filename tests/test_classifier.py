import pytest
import torch
import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import HalluClassifier

def test_classifier_init():
    input_size = 8192
    model = HalluClassifier(input_size=input_size)
    assert isinstance(model, torch.nn.Module)
    
    # Check if the first layer has correct input size
    first_layer = model.net[1] # net[0] is Dropout
    assert first_layer.in_features == input_size

def test_classifier_forward():
    input_size = 4096
    batch_size = 8
    model = HalluClassifier(input_size=input_size)
    
    x = torch.randn(batch_size, input_size)
    output = model(x)
    
    assert output.shape == (batch_size, 2)
    assert not torch.isnan(output).any()

def test_classifier_dropout():
    # Verify dropout value
    model = HalluClassifier(input_size=10, dropout=0.5)
    assert model.net[0].p == 0.5
