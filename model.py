"""Baseline MLP for Purchase-100.

Architecture matches Shokri et al. 2017: a simple feedforward network
with two hidden layers. Kept deliberately small and unregularized so it
overfits — overfitting is what makes MIA work, and what DP-SGD must defend.
"""

import torch
import torch.nn as nn


class PurchaseMLP(nn.Module):
    def __init__(
            self,
            input_dim: int = 600,
            hidden_dim: int = 128,
            num_classes: int = 100,
            activation: str = "tanh",
    ):
        super().__init__()

        act_layer = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits. Apply softmax / cross-entropy externally.
        return self.net(x)


def build_model(config: dict) -> PurchaseMLP:
    """Construct a model from a config dict.

    Expected keys (with defaults): input_dim, hidden_dim, num_classes, activation.
    """
    return PurchaseMLP(
        input_dim=config.get("input_dim", 600),
        hidden_dim=config.get("hidden_dim", 128),
        num_classes=config.get("num_classes", 100),
        activation=config.get("activation", "tanh"),
    )