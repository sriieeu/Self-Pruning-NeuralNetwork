"""
visualize.py — Gate Distribution & Training Visualisations
============================================================
Generates three plot types:

1. gate_distribution  — histogram of final gate values for one λ run.
                        Successful pruning shows a spike at 0 (pruned)
                        and a cluster near 1 (active).

2. training_curves    — accuracy, sparsity, and loss over epochs for
                        all λ values side-by-side.

3. lambda_comparison  — grouped bar chart: test accuracy vs sparsity
                        across all λ values.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── shared style ──────────────────────────────────────────────────────

BG_DARK  = "#0d1117"
BG_PANEL = "#161b22"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
BLUE     = "#58a6ff"
ORANGE   = "#f0883e"
GREEN    = "#3fb950"
PURPLE   = "#bc8cff"
PALETTE  = [BLUE, GREEN, ORANGE, PURPLE]


def _apply_dark_style(fig, ax_list):
    """Apply a consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(BG_DARK)
    for ax in (ax_list if isinstance(ax_list, (list, tuple)) else [ax_list]):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)


# ── 1. Gate distribution histogram ────────────────────────────────────

def plot_gate_distribution(result: dict, save_path: str) -> None:
    """
    Histogram of final gate values.

    Bars near 0 are coloured orange (pruned neurons).
    Bars elsewhere are blue (active neurons).

    Parameters
    ----------
    result    : dict returned by train.train()
    save_path : output PNG path
    """
    gate_vals = result["gate_values"]
    lam       = result["lam"]
    sparsity  = result["sparsity"]
    acc       = result["test_acc"]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(fig, ax)

    counts, bins, patches = ax.hist(
        gate_vals, bins=30, edgecolor=BG_DARK, linewidth=0.5, alpha=0.88
    )
    for patch, left_edge in zip(patches, bins[:-1]):
        patch.set_facecolor(ORANGE if left_edge < 0.05 else BLUE)

    # annotate pruned spike
    n_pruned = int((gate_vals < 1e-2).sum())
    if counts.max() > 0:
        ax.annotate(
            f"{n_pruned} neurons pruned\n(gate < 0.01)",
            xy=(0.02, counts.max()),
            xytext=(0.28, counts.max() * 0.80),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
            color=ORANGE, fontsize=10,
        )

    ax.set_xlabel("Gate value", color=MUTED, fontsize=12, labelpad=8)
    ax.set_ylabel("Neuron count", color=MUTED, fontsize=12, labelpad=8)
    ax.set_title(
        f"Hard Concrete gate distribution   λ={lam:.0e}\n"
        f"Sparsity {sparsity:.2%}  ·  Test acc {acc:.3f}",
        color=TEXT, fontsize=13, pad=12,
    )

    # legend patches
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=ORANGE, label="Pruned  (gate ≈ 0)"),
            Patch(color=BLUE,   label="Active  (gate ≈ 1)"),
        ],
        facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# ── 2. Training curves ────────────────────────────────────────────────

def plot_training_curves(results: list, save_path: str) -> None:
    """
    One row per λ value, three columns: test accuracy / sparsity / total loss.

    Parameters
    ----------
    results   : list of dicts returned by train.train()
    save_path : output PNG path
    """
    n   = len(results)
    fig = plt.figure(figsize=(14, 4 * n))
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.55, wspace=0.35)

    metrics = [
        ("test_acc",   "Test accuracy"),
        ("sparsity",   "Sparsity level"),
        ("total_loss", "Total loss"),
    ]

    for row, res in enumerate(results):
        h   = res["history"]
        lam = res["lam"]
        c   = PALETTE[row % len(PALETTE)]
        for col, (key, label) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])
            _apply_dark_style(fig, ax)
            ax.plot(h[key], color=c, linewidth=1.8)
            ax.set_title(
                f"λ={lam:.0e}  ·  {label}", color=TEXT, fontsize=11, pad=6
            )
            ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
            ax.tick_params(labelsize=9)

    fig.suptitle(
        "Training dynamics — self-pruning network",
        color=TEXT, fontsize=15, y=1.01,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# ── 3. Lambda trade-off comparison ────────────────────────────────────

def plot_lambda_comparison(results: list, save_path: str) -> None:
    """
    Grouped bar chart: accuracy (blue) and sparsity (orange) for each λ.

    Parameters
    ----------
    results   : list of dicts returned by train.train()
    save_path : output PNG path
    """
    labels     = [f"λ={r['lam']:.0e}" for r in results]
    accs       = [r["test_acc"] * 100   for r in results]
    sparsities = [r["sparsity"]  * 100  for r in results]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(fig, ax)

    b1 = ax.bar(x - w / 2, accs,       w, label="Test accuracy (%)",  color=BLUE,   alpha=0.85)
    b2 = ax.bar(x + w / 2, sparsities, w, label="Sparsity level (%)", color=ORANGE, alpha=0.85)
    ax.bar_label(b1, fmt="%.1f", color=TEXT, fontsize=9, padding=3)
    ax.bar_label(b2, fmt="%.1f", color=TEXT, fontsize=9, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=MUTED, fontsize=11)
    ax.set_ylabel("%", color=MUTED, fontsize=12)
    ax.set_title("λ trade-off: accuracy vs sparsity", color=TEXT, fontsize=13, pad=12)
    ax.legend(facecolor=BG_PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# ── Convenience wrapper ────────────────────────────────────────────────

def save_all_plots(results: list, output_dir: str) -> None:
    """
    Save all three plot types for a list of training results.

    Parameters
    ----------
    results    : list of dicts returned by train.train()
    output_dir : directory to write PNG files
    """
    os.makedirs(output_dir, exist_ok=True)

    for res in results:
        plot_gate_distribution(
            res,
            save_path=os.path.join(output_dir, f"gate_dist_lam{res['lam']:.0e}.png"),
        )

    plot_training_curves(
        results,
        save_path=os.path.join(output_dir, "training_curves.png"),
    )
    plot_lambda_comparison(
        results,
        save_path=os.path.join(output_dir, "lambda_comparison.png"),
    )
