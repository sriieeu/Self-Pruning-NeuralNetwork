"""
train.py — Training and Evaluation Loop
========================================
Implements the custom training loop for the self-pruning network.

Total Loss = CrossEntropyLoss + λ × (L0_penalty / total_neurons)

The L0 penalty is normalised by the total number of gated neurons so that
λ is scale-invariant across different network architectures.
"""

import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import get_cifar10_loaders
from model import SelfPruningNet


# ------------------------------------------------------------------
# One epoch of training
# ------------------------------------------------------------------

def train_one_epoch(
    model: SelfPruningNet,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    lam: float,
) -> dict:
    """
    Run one full pass over the training set.

    Returns
    -------
    dict with keys: total_loss, ce_loss, l0_loss, train_acc
    """
    model.train()
    total_loss = ce_sum = l0_sum = 0.0
    correct = total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)

        ce_loss  = F.cross_entropy(logits, y)
        l0_pen   = model.l0_penalty() / model.total_neurons()   # normalised
        loss     = ce_loss + lam * l0_pen

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        ce_sum     += ce_loss.item()
        l0_sum     += l0_pen.item()
        pred        = logits.argmax(dim=1)
        correct    += (pred == y).sum().item()
        total      += y.size(0)

    n = len(loader)
    return {
        "total_loss": total_loss / n,
        "ce_loss":    ce_sum     / n,
        "l0_loss":    l0_sum     / n,
        "train_acc":  correct    / total,
    }


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: SelfPruningNet, loader, device: str) -> float:
    """Return top-1 accuracy on *loader*."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ------------------------------------------------------------------
# Full training run for one λ value
# ------------------------------------------------------------------

def train(
    lam: float,
    epochs: int = 30,
    temperature: float = 2 / 3,
    device: str = "cpu",
    data_dir: str = "./data",
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    """
    Train the self-pruning network for a single λ value.

    Parameters
    ----------
    lam         : sparsity regularisation weight
    epochs      : number of training epochs
    temperature : Hard Concrete β  (lower → more binary gates)
    device      : 'cuda' or 'cpu'
    data_dir    : path to download/cache CIFAR-10
    batch_size  : mini-batch size
    lr          : initial Adam learning rate

    Returns
    -------
    dict containing final metrics, per-layer sparsities, gate values,
    and full training history.
    """
    print(f"\n{'='*60}")
    print(f"  λ={lam:.0e}  β={temperature}  epochs={epochs}  lr={lr}")
    print(f"{'='*60}")

    train_ld, test_ld = get_cifar10_loaders(batch_size, data_dir)
    model             = SelfPruningNet(temperature=temperature).to(device)
    optimizer         = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler         = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        t0       = time.time()
        metrics  = train_one_epoch(model, train_ld, optimizer, device, lam)
        test_acc = evaluate(model, test_ld, device)
        scheduler.step()

        model.eval()
        sparsity = model.sparsity_level()

        for k, v in metrics.items():
            history[k].append(v)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Loss {metrics['total_loss']:.4f} | "
            f"CE {metrics['ce_loss']:.4f} | "
            f"L0 {metrics['l0_loss']:.4f} | "
            f"Train {metrics['train_acc']:.3f} | "
            f"Test {test_acc:.3f} | "
            f"Sparsity {sparsity:.2%} | "
            f"{time.time() - t0:.1f}s"
        )

    model.eval()
    final_acc      = evaluate(model, test_ld, device)
    final_sparsity = model.sparsity_level()
    layer_sp       = model.layer_sparsities()
    gate_vals      = model.all_gate_values()

    print(f"\n  ── Final  λ={lam:.0e} ──────────────────────────")
    print(f"  Test Accuracy  : {final_acc:.4f}")
    print(f"  Sparsity Level : {final_sparsity:.2%}")
    for k, v in layer_sp.items():
        print(f"    {k}: {v:.2%} pruned")

    # Save checkpoint so the inference server can load it
    import os
    ckpt_dir = data_dir.replace("./data", "./outputs") if "./data" in data_dir else "./outputs"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"model_lam{lam:.0e}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "lam":              lam,
        "temperature":      temperature,
        "test_acc":         final_acc,
        "sparsity":         final_sparsity,
        "layer_sparsity":   layer_sp,
    }, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    return {
        "lam":            lam,
        "temperature":    temperature,
        "test_acc":       final_acc,
        "sparsity":       final_sparsity,
        "layer_sparsity": layer_sp,
        "gate_values":    gate_vals,
        "history":        dict(history),
    }

