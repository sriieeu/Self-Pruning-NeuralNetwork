# Self-Pruning Neural Network
### Hard Concrete Gates · Neuron-Level Pruning · CIFAR-10

A feed-forward network that **learns to prune itself during training** using the Hard Concrete gate mechanism (Louizos et al., 2018). No post-training step — neurons are removed on-the-fly through a differentiable L0 regularisation loss.

---

## Quick Start

```bash
# Linux / macOS
bash setup.sh --run

# Windows
setup.bat --run
```

That single command creates the venv, installs all dependencies, and launches training with default settings.

---

## Project Structure

```
self-pruning-nn/
├── gates.py          # HardConcreteGate — BinaryConcrete sampling & L0 penalty
├── layers.py         # PrunableLinear   — neuron-level gating layer
├── model.py          # SelfPruningNet   — full CIFAR-10 feed-forward network
├── data.py           # get_cifar10_loaders — train/test splits with augmentation
├── train.py          # train(), evaluate() — custom training loop
├── visualize.py      # Gate histograms, training curves, λ comparison plots
├── report.py         # Markdown + JSON report generation
├── main.py           # CLI entry point
├── requirements.txt  # Python dependencies
├── setup.sh          # Linux/macOS one-shot setup script
├── setup.bat         # Windows one-shot setup script
└── README.md
```

---

## Manual Setup (step by step)

```bash
# 1. Create venv
python -m venv venv

# 2. Activate
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate.bat      # Windows

# 3. Install
pip install -r requirements.txt

# 4. Run
python main.py
```

---

## Training Options

```bash
python main.py [OPTIONS]

Options:
  --lambdas     FLOATS   λ values to compare  (default: 1e-4 1e-3 5e-3)
  --epochs      INT      Epochs per run        (default: 30)
  --temperature FLOAT    Hard Concrete β       (default: 0.667)
  --batch_size  INT      Mini-batch size       (default: 256)
  --lr          FLOAT    Adam learning rate    (default: 1e-3)
  --data_dir    PATH     CIFAR-10 cache        (default: ./data)
  --output_dir  PATH     Plots & reports       (default: ./outputs)
```

Example — aggressive comparison across five λ values for 50 epochs:

```bash
python main.py --lambdas 1e-5 1e-4 1e-3 5e-3 1e-2 --epochs 50 --temperature 0.5
```

---

## Method

### Hard Concrete Gate

Each output neuron gets a learnable `log_α` parameter. During training a stochastic gate is drawn from the BinaryConcrete distribution:

```
u      ~ Uniform(0, 1)
s      = σ( (log u − log(1−u) + log_α) / β )   # temperature β
s_bar  = s · (ζ − γ) + γ                         # stretch to (γ, ζ)
gate   = clamp(s_bar, 0, 1)                       # hard clamp → mass at 0 and 1
```

Parameters `ζ = 1.1`, `γ = −0.1` push probability mass to **exactly 0 and 1**, enabling genuinely sparse solutions. At inference time: `gate = 1[σ(log_α) > 0.5]`.

### Loss Function

```
Total Loss = CrossEntropy + λ × (L0_penalty / total_neurons)

L0_penalty = Σ_i  σ( log_α_i − β · log(−γ/ζ) )
```

### Architecture

```
Input (3072)
  └─ PrunableLinear(3072 → 1024) + BatchNorm + ReLU
  └─ PrunableLinear(1024 →  512) + BatchNorm + ReLU
  └─ PrunableLinear( 512 →  256) + BatchNorm + ReLU
  └─ Linear(256 → 10)              ← no gate on classifier head
```

Total gated neurons: **1792** (1024 + 512 + 256)

### Temperature β

| β | Effect |
|---|--------|
| 0.1 – 0.3 | Nearly binary gates from the start; aggressive pruning |
| 0.5 – 0.7 | Balanced (paper default: 2/3 ≈ 0.667) |
| 0.8 – 1.0 | Soft gates; smoother gradients; slower convergence to sparse |

---

## λ Selection Guide

| λ | Profile | ~Accuracy | ~Sparsity |
|---|---------|-----------|-----------|
| `1e-5` | Near-dense baseline | 56 % | 10 % |
| `1e-4` | Light pruning | 54 % | 30 % |
| `1e-3` | **Balanced (recommended)** | 51 % | 60 % |
| `5e-3` | Aggressive | 47 % | 80 % |
| `1e-2` | Maximum sparsity | 43 % | 92 % |

---

## Outputs

After training, `./outputs/` contains:

| File | Description |
|------|-------------|
| `gate_dist_lam*.png` | Gate histogram per λ — spike at 0 = pruned, cluster near 1 = active |
| `training_curves.png` | Accuracy / sparsity / loss curves for all λ values |
| `lambda_comparison.png` | Grouped bar chart: accuracy vs sparsity across λ values |
| `report.md` | Full analysis with results table |
| `results.json` | Machine-readable summary (gate values, history, metrics) |

---

## Reference

> Louizos, C., Welling, M., & Kingma, D. P. (2018).
> *Learning Sparse Neural Networks through L0 Regularization.*
> ICLR 2018. https://arxiv.org/abs/1712.01312
