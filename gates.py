"""
gates.py — Hard Concrete Gate (Louizos et al., 2018)
=====================================================
Implements the BinaryConcrete relaxation used for differentiable L0 regularisation.

Reference:
    Louizos, C., Welling, M., & Kingma, D. P. (2018).
    Learning Sparse Neural Networks through L0 Regularization.
    ICLR 2018. https://arxiv.org/abs/1712.01312
"""

import numpy as np
import torch
import torch.nn as nn


class HardConcreteGate(nn.Module):
    """
    Hard Concrete gate for a single layer (neuron-level pruning).

    Each output neuron gets ONE learnable log-alpha (log_α) parameter.

    During training
    ---------------
    A "stretched" sigmoid samples a soft gate in (0, 1):

        u      ~ Uniform(0, 1)
        s      = σ( (log u − log(1−u) + log_α) / β )
        s_bar  = s * (ζ − γ) + γ          # stretch to (γ, ζ)
        gate   = clamp(s_bar, 0, 1)        # hard clamp → mass at 0 and 1

    During evaluation
    -----------------
    The gate is snapped to a hard {0, 1} value:

        gate = 1[σ(log_α) > 0.5]

    Parameters
    ----------
    n_neurons   : int   — number of output neurons to gate
    temperature : float — β in the BinaryConcrete relaxation
                          lower β  →  more binary gates  →  more aggressive pruning
    zeta        : float — upper stretch bound (ζ > 1, default 1.1 from paper)
    gamma       : float — lower stretch bound (γ < 0, default −0.1 from paper)
    """

    def __init__(
        self,
        n_neurons: int,
        temperature: float = 2 / 3,
        zeta: float = 1.1,
        gamma: float = -0.1,
    ):
        super().__init__()
        self.n_neurons   = n_neurons
        self.temperature = temperature
        self.zeta        = zeta
        self.gamma       = gamma

        # one learnable scalar per output neuron, initialised at 0
        self.log_alpha = nn.Parameter(torch.zeros(n_neurons))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        """Return gate values: soft during training, hard {0,1} at eval."""
        if self.training:
            return self._sample_gate()
        return self._deterministic_gate()

    def _sample_gate(self) -> torch.Tensor:
        """
        Reparameterised sample — gradients flow through log_alpha.
        """
        u = (
            torch.zeros_like(self.log_alpha)
            .uniform_()
            .clamp(1e-8, 1.0 - 1e-8)
        )
        s = torch.sigmoid(
            (torch.log(u) - torch.log(1.0 - u) + self.log_alpha)
            / self.temperature
        )
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0.0, 1.0)

    def _deterministic_gate(self) -> torch.Tensor:
        """Hard binary gate — no stochasticity, used during inference."""
        return (torch.sigmoid(self.log_alpha) > 0.5).float()

    # ------------------------------------------------------------------
    # L0 Penalty
    # ------------------------------------------------------------------

    def l0_penalty(self) -> torch.Tensor:
        """
        Expected number of non-zero gates (differentiable).

        Minimising this term drives log_alpha downward, pushing gates to 0.

        Formula (from Louizos et al.):
            Σ_i  σ( log_α_i − β · log(−γ / ζ) )
        """
        return torch.sigmoid(
            self.log_alpha
            - self.temperature * np.log(-self.gamma / self.zeta)
        ).sum()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gate_values(self) -> np.ndarray:
        """Deterministic {0, 1} gate array for post-training analysis."""
        return self._deterministic_gate().cpu().numpy()

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of neurons with gate value below *threshold*."""
        g = self._sample_gate() if self.training else self._deterministic_gate()
        return (g < threshold).float().mean().item()
