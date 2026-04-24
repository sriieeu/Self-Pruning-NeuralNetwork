"""
infer.py — Standalone Inference for Self-Pruning NN
=====================================================
Loads a saved model checkpoint (produced by train.py) and runs
inference on a single image.

Usage (CLI)
-----------
    python infer.py --image path/to/img.jpg
    python infer.py --image path/to/img.jpg --checkpoint outputs/model_lam1e-03.pt

Usage (API — imported by server.py)
------------------------------------
    from infer import load_model, predict
    model, meta = load_model("outputs/model_lam1e-03.pt")
    result = predict(model, pil_image)
"""

import os
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from model import SelfPruningNet


# ── CIFAR-10 labels ────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_EMOJIS = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚛"]

# CIFAR-10 normalisation statistics (training-set mean / std)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

# ── Pre-processing pipeline ────────────────────────────────────────────
def get_transform() -> T.Compose:
    """Return the same normalisation used during CIFAR-10 training."""
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


# ── Find the best available checkpoint ────────────────────────────────
def find_latest_checkpoint(output_dir: str = "./outputs") -> str | None:
    """Return path of the checkpoint with the lowest λ (densest / best acc)."""
    pattern = os.path.join(output_dir, "model_lam*.pt")
    ckpts = sorted(glob.glob(pattern))
    return ckpts[0] if ckpts else None


# ── Load model from checkpoint ─────────────────────────────────────────
def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[SelfPruningNet, dict]:
    """
    Load a SelfPruningNet from a .pt checkpoint produced by train.py.

    Returns
    -------
    model : SelfPruningNet  (eval mode, on *device*)
    meta  : dict with lambda, temperature, test_acc, sparsity, layer_sparsity
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    temperature = ckpt.get("temperature", 2 / 3)

    model = SelfPruningNet(temperature=temperature)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "checkpoint":    checkpoint_path,
        "lambda":        ckpt.get("lam", None),
        "temperature":   temperature,
        "test_acc":      ckpt.get("test_acc", None),
        "sparsity":      ckpt.get("sparsity", None),
        "layer_sparsity":ckpt.get("layer_sparsity", {}),
    }
    return model, meta


# ── Single-image inference ─────────────────────────────────────────────
@torch.no_grad()
def predict(
    model: SelfPruningNet,
    image: Image.Image,
    device: str = "cpu",
) -> dict:
    """
    Run inference on a PIL Image.

    Parameters
    ----------
    model  : loaded SelfPruningNet (eval mode)
    image  : PIL.Image.Image — any size / mode
    device : 'cpu' or 'cuda'

    Returns
    -------
    dict with:
        predicted_class  : str
        predicted_emoji  : str
        confidence       : float  (0-1)
        probabilities    : list[dict]  — [{class, emoji, prob}, ...]
        active_neurons   : int
        pruned_neurons   : int
        sparsity         : float
        gate_values      : list[float]
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)   # (1, 3, 32, 32)

    logits = model(tensor)                              # (1, 10)
    probs  = F.softmax(logits, dim=1).squeeze().tolist()

    top_idx   = int(torch.argmax(logits).item())
    gate_vals = model.all_gate_values().tolist()
    total     = len(gate_vals)
    pruned    = int(sum(1 for g in gate_vals if g < 0.01))
    active    = total - pruned

    return {
        "predicted_class": CIFAR10_CLASSES[top_idx],
        "predicted_emoji": CIFAR10_EMOJIS[top_idx],
        "confidence":      round(probs[top_idx], 4),
        "probabilities": [
            {
                "class": CIFAR10_CLASSES[i],
                "emoji": CIFAR10_EMOJIS[i],
                "prob":  round(probs[i], 4),
            }
            for i in range(10)
        ],
        "active_neurons": active,
        "pruned_neurons": pruned,
        "sparsity":       round(pruned / total, 4),
        "gate_values":    gate_vals,
    }


# ── CLI entry point ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run single-image inference with a trained Self-Pruning NN"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image (any format PIL supports)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the .pt checkpoint (default: auto-detect in ./outputs/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to search for checkpoints (default: ./outputs)",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.output_dir)
    if ckpt_path is None:
        print("❌  No checkpoint found. Run  python main.py  first to train and save a model.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint : {ckpt_path}")
    model, meta = load_model(ckpt_path, device)
    print(f"  λ={meta['lambda']:.0e}  |  Test acc={meta['test_acc']:.4f}"
          f"  |  Sparsity={meta['sparsity']:.2%}")

    image = Image.open(args.image)
    result = predict(model, image, device)

    print(f"\nPrediction : {result['predicted_emoji']}  {result['predicted_class'].upper()}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"Active neurons : {result['active_neurons']} / {result['active_neurons'] + result['pruned_neurons']}")
    print(f"Sparsity       : {result['sparsity']:.2%}")
    print("\nAll class probabilities:")
    for p in sorted(result["probabilities"], key=lambda x: -x["prob"]):
        bar = "█" * int(p["prob"] * 30)
        print(f"  {p['emoji']} {p['class']:12s}  {p['prob']:6.2%}  {bar}")


if __name__ == "__main__":
    main()
