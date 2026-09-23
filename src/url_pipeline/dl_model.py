"""
Lightweight Tabular Deep Neural Network for URL Classification.

Architecture:
  - Compact PyTorch Feedforward / Residual Network (HybridURLNet)
  - Designed for hybrid URL features (lexical, host, structural, content, statistical)
  - Under 50,000 parameters (< 200 KB in memory/disk)
  - Built-in support for INT8 dynamic quantization for ultra-low latency & green inference.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout for tabular data."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act1(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        out = self.act2(out + residual)
        return out


class HybridURLNet(nn.Module):
    """
    Lightweight Tabular Deep Learning Network for multi-class URL classification.
    
    Classes:
      0: benign
      1: phishing
      2: malware
      3: defacement
    """

    def __init__(
        self,
        input_dim: int = 45,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        # Non-linear representation learning via residual connection
        self.res_block = ResidualBlock(hidden_dim, dropout_rate=dropout_rate)

        # Bottleneck and classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits."""
        # Ensure 2D tensor (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.input_layer(x)
        h = self.res_block(h)
        logits = self.classifier(h)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax class probabilities."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns argmax predicted class indices."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def quantize(self) -> nn.Module:
        """
        Dynamically quantizes linear layers to INT8 for minimal latency,
        smaller memory footprint, and lower energy consumption at inference.
        """
        self.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            self.to("cpu"),
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model


def save_checkpoint(
    model: HybridURLNet,
    scaler_state: Optional[Dict[str, Any]],
    feature_names: list[str],
    label_mapping: Dict[int, str],
    save_path: Union[str, Path],
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves model weights along with scaler parameters, feature schema, and label mappings."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_classes": model.num_classes,
        "param_count": model.count_parameters(),
        "scaler": scaler_state,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
        "metrics": extra_metrics or {},
    }
    torch.save(payload, str(save_path))


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
) -> tuple[HybridURLNet, Dict[str, Any]]:
    """Loads a saved checkpoint and returns the instantiated model and metadata."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = HybridURLNet(
        input_dim=checkpoint.get("input_dim", 45),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_classes=checkpoint.get("num_classes", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
