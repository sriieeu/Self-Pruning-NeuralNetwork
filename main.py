"""
main.py — Entry Point
======================
Trains the self-pruning network for each λ value, then saves all
plots and reports.

Usage
-----
    python main.py                                # defaults
    python main.py --lambdas 1e-4 1e-3 5e-3     # custom λ values
    python main.py --epochs 50 --temperature 0.5
    python main.py --help                         # full option list
"""

import argparse
import os
import torch

from train import train
from visualize import save_all_plots
from report import save_reports


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-Pruning Neural Network with Hard Concrete Gates"
    )
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=[1e-4, 1e-3, 5e-3],
        help="Sparsity regularisation weights to compare (default: 1e-4 1e-3 5e-3)",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs per λ run (default: 30)",
    )
    p.add_argument(
        "--temperature", type=float, default=2 / 3,
        help="Hard Concrete temperature β — lower = more binary gates (default: 0.667)",
    )
    p.add_argument(
        "--batch_size", type=int, default=256,
        help="Mini-batch size (default: 256)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial Adam learning rate (default: 1e-3)",
    )
    p.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory to cache CIFAR-10 (default: ./data)",
    )
    p.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory for plots and reports (default: ./outputs)",
    )
    return p.parse_args()


def print_summary(results: list) -> None:
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Lambda':>10} | {'Test Acc':>10} | {'Sparsity':>10}")
    print("-" * 40)
    for r in results:
        print(
            f"  {r['lam']:>10.0e} | "
            f"{r['test_acc']:>10.4f} | "
            f"{r['sparsity']:>10.2%}"
        )


def main() -> None:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")
    print(f"λ values: {args.lambdas}")
    print(f"Epochs  : {args.epochs}")
    print(f"Temp β  : {args.temperature}")

    results = []
    for lam in args.lambdas:
        result = train(
            lam=lam,
            epochs=args.epochs,
            temperature=args.temperature,
            device=device,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        results.append(result)

    print_summary(results)

    os.makedirs(args.output_dir, exist_ok=True)
    save_all_plots(results, args.output_dir)
    save_reports(results, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
