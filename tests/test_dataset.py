import pytest
import torch
import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import TrainDataset

class MockArgs:
    pass

def test_dataset_len():
    train_data = [
        {"label": 0, "hd": [0.1, 0.2]},
        {"label": 1, "hd": [0.3, 0.4]},
        {"label": 0, "hd": [0.5, 0.6]}
    ]
    args = MockArgs()
    dataset = TrainDataset(train_data, args)
    
    assert len(dataset) == 3
    assert dataset.halu_num == 1

def test_dataset_getitem():
    train_data = [
        {"label": 1, "hd": [1.0, 2.0, 3.0]}
    ]
    args = MockArgs()
    dataset = TrainDataset(train_data, args)
    
    sample = dataset[0]
    
    assert "input" in sample
    assert "y" in sample
    assert torch.equal(sample["input"], torch.tensor([1.0, 2.0, 3.0]))
    assert sample["y"].item() == 1
    assert sample["y"].shape == (1,)
