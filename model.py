"""
model.py — Self-Pruning Feed-Forward Network for CIFAR-10
==========================================================
Architecture:
    3072  →  PrunableLinear(1024)  →  BN  →  ReLU
          →  PrunableLinear(512)   →  BN  →  ReLU
          →  PrunableLinear(256)   →  BN  →  ReLU
          →  Linear(10)                     (no gate on classifier head)

Total gated neurons: 1024 + 512 + 256 = 1792
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    Feed-forward network with Hard Concrete neuron-level pruning.

    Parameters
    ----------
    temperature : float
        Hard Concrete temperature β.  Lower values push gates toward
        binary {0, 1} faster (more aggressive pruning during training).
        Typical range: 0.1 – 1.0.  Paper default: 2/3.
    """

    # hidden-layer sizes — change here to experiment with other architectures
    HIDDEN = [1024, 512, 256]

    def __init__(self, temperature: float = 2 / 3):
        super().__init__()
        self.flatten = nn.Flatten()

        dims = [3072] + self.HIDDEN
        self.prunable_layers = nn.ModuleList([
            PrunableLinear(dims[i], dims[i + 1], temperature)
            for i in range(len(self.HIDDEN))
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(h) for h in self.HIDDEN
        ])

        # final classifier — not gated (always kept)
        self.classifier = nn.Linear(self.HIDDEN[-1], 10)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for layer, bn in zip(self.prunable_layers, self.batch_norms):
            x = F.relu(bn(layer(x)))
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def l0_penalty(self) -> torch.Tensor:
        """Sum of L0 penalties across all prunable layers."""
        return sum(layer.l0_penalty() for layer in self.prunable_layers)

    def total_neurons(self) -> int:
        """Total number of gated neurons in the network."""
        return sum(layer.gate.n_neurons for layer in self.prunable_layers)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def all_gate_values(self) -> np.ndarray:
        """
        Concatenated deterministic {0, 1} gate values across all layers.
        Useful for plotting the gate distribution histogram.
        """
        return np.concatenate([
            layer.gate.gate_values() for layer in self.prunable_layers
        ])

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Global sparsity: fraction of neurons whose gate value < threshold.

        A value of 0.80 means 80 % of neurons have been pruned.
        """
        vals = self.all_gate_values()
        return float((vals < threshold).sum() / len(vals))

    def layer_sparsities(self, threshold: float = 1e-2) -> dict:
        """Per-layer sparsity dictionary, e.g. {'layer_1': 0.72, ...}."""
        result = {}
        for i, layer in enumerate(self.prunable_layers):
            g = layer.gate.gate_values()
            result[f"layer_{i + 1}"] = float((g < threshold).mean())
        return result
