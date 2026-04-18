import torch.nn as nn


class HalluClassifier(nn.Module):
    """4-layer MLP: classifies a hidden-state vector as correct vs hallucinated."""

    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)
