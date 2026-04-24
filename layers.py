"""
layers.py — Prunable Linear Layer
==================================
A drop-in replacement for torch.nn.Linear that gates each output
neuron with a Hard Concrete gate (Louizos et al., 2018).

Usage
-----
    from layers import PrunableLinear
    layer = PrunableLinear(in_features=512, out_features=256)
    out   = layer(x)         # (batch, 256) — some neurons will be zeroed
    pen   = layer.l0_penalty()  # scalar — add * λ to your loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gates import HardConcreteGate


class PrunableLinear(nn.Module):
    """
    Linear layer with a learnable Hard Concrete gate per OUTPUT neuron.

    Forward pass
    ------------
        h      = x @ weight.T + bias          # standard linear transform
        gates  = HardConcreteGate()           # shape: (out_features,)
        output = h * gates                    # element-wise broadcast over batch

    Because gating happens at the neuron level (one gate per output
    neuron rather than one gate per weight), entire rows of the weight
    matrix are pruned simultaneously.  This is equivalent to removing
    a neuron from the network entirely.

    Parameters
    ----------
    in_features  : int   — input dimension
    out_features : int   — output dimension (number of gated neurons)
    temperature  : float — Hard Concrete temperature β (passed to the gate)
    bias         : bool  — whether to include a bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        temperature: float = 2 / 3,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # standard learnable parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # one Hard Concrete gate per output neuron
        self.gate = HardConcreteGate(out_features, temperature=temperature)

        # kaiming initialisation for ReLU networks
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h     = F.linear(x, self.weight, self.bias)   # (batch, out_features)
        gates = self.gate()                            # (out_features,)
        return h * gates                               # broadcast over batch dim

    def l0_penalty(self) -> torch.Tensor:
        """Delegate L0 penalty to the underlying gate."""
        return self.gate.l0_penalty()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"temperature={self.gate.temperature}"
        )
