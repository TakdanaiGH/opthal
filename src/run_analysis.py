#!/usr/bin/env python3
"""
run_analysis.py
===============
Post-training analysis runner for the Cosine Prototypical Network.

Loads a trained model checkpoint and runs:
  1. EmbeddingVisualizer  – embedding space scatter plots per episode
  2. compare_hd_vs_2d_predictions – HD vs 2D distance comparison

Reads model architecture (backbone, embedding_dims) from meta_experiments.csv
using the experiment ID extracted from the model filename.

Configurable via CLI (or via run_analysis.sh):
  --model               path to model checkpoint, e.g. ../models/model_0f46a3ef.pth
  --max_episode          number of episodes to visualize (default: 10)
  --reduction_method     tsne | umap (default: tsne)
  --jitter_strength      jitter for scatter plots (default: 0.02)

All output files are saved under analysis/{exp_id}/:
  analysis/{exp_id}/episode_{n}.svg
  analysis/{exp_id}/comparison.csv
"""

import argparse
import ast
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

# --- project modules ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module.preprocess import ImageRegistryBuilder
from module.config import RuntimeConfig
from module.dataspliter import DataSplitter
from module.dataset import ImageDataset
from module.transform import ImageTransforms
from module.model import PrototypicalNetwork
from module.visualizer import EmbeddingVisualizer
from module.comparison import compare_hd_vs_2d_predictions


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Run post-training analysis on a saved model")
    p.add_argument("--model", type=str, required=True,
                   help="Path to model checkpoint, e.g. ../models/model_0f46a3ef.pth")
    p.add_argument("--max_episode", type=int, default=10,
                   help="Number of episodes to visualize (default: 10)")
    p.add_argument("--reduction_method", type=str, default="tsne",
                   choices=["tsne", "umap"],
                   help="Dimensionality reduction method (default: tsne)")
    p.add_argument("--jitter_strength", type=float, default=0.02,
                   help="Jitter strength for scatter plots (default: 0.02)")
    return p.parse_args()


# ============================================================
# Helpers
# ============================================================
def extract_exp_id(model_path: str) -> str:
    """Extract experiment ID from model filename like model_0f46a3ef.pth."""
    stem = Path(model_path).stem  # model_0f46a3ef
    if stem.startswith("model_"):
        return stem[len("model_"):]
    return stem


def lookup_experiment(exp_id: str, meta_path: Path) -> dict:
    """Look up experiment metadata from meta_experiments.csv."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta experiments file not found: {meta_path}")

    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["experiment_id"] == exp_id:
                return row

    raise ValueError(
        f"Experiment ID '{exp_id}' not found in {meta_path}.\n"
        f"Make sure the model was trained with run_experiment.py."
    )


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(os.path.dirname(os.path.abspath(__file__))) / model_path

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    exp_id = extract_exp_id(str(model_path))
    meta_path = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "meta_experiments.csv"
    meta = lookup_experiment(exp_id, meta_path)

    backbone = meta["backbone"]
    embedding_dims = tuple(ast.literal_eval(meta["embedding_dims"]))

    print(f"\n{'#'*70}")
    print(f"  ANALYSIS  {exp_id}")
    print(f"{'#'*70}")
    print(f"  model file             = {model_path}")
    print(f"  backbone               = {backbone}")
    print(f"  embedding_dims         = {embedding_dims}")
    print(f"  max_episode            = {args.max_episode}")
    print(f"  reduction_method       = {args.reduction_method}")
    print(f"  jitter_strength        = {args.jitter_strength}")
    print(f"{'#'*70}\n")

    # ---- runtime ----
    runtime = RuntimeConfig(seed=42, verbose=True)
    device = runtime.setup()

    # ---- image registry ----
    builder = ImageRegistryBuilder(root_dir="../images")
    builder.scan_directory()
    builder.save_to_csv(output_path="../dataset.csv", use_relative_paths=True)

    # ---- split ----
    splitter = DataSplitter(
        csv_path="../dataset.csv",
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42,
    )
    train_df, val_df, test_df = splitter.split_data()

    # ---- transforms ----
    tx = ImageTransforms(image_size=224, augment_training=False)

    # ---- datasets ----
    train_ds = ImageDataset(train_df, root_dir="..",
                            transform=tx.get_test_transform())
    test_ds = ImageDataset(test_df, root_dir="..",
                           transform=tx.get_test_transform(),
                           label_encoder=train_ds.get_label_encoder())
    print(f"Datasets  train={len(train_ds)}  test={len(test_ds)}")

    # ---- model ----
    model = PrototypicalNetwork(
        backbone=backbone,
        embedding_dims=embedding_dims,
        pretrained=True,
        cache_dir="../backbones",
    )
    model.freeze_backbone()

    # ---- load checkpoint ----
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.print_model_info()

    n_way = ckpt.get("n_way", 2)
    k_shot = ckpt.get("k_shot", 3)

    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
          f"val_loss={ckpt.get('val_loss', 'N/A')}")
    print(f"Episode config: {n_way}-way {k_shot}-shot\n")

    # ---- output directory: analysis/{exp_id}/ ----
    analysis_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "analysis" / exp_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {analysis_dir}\n")

    # ================================================================
    # 1. Embedding Visualization
    # ================================================================
    print("=" * 70)
    print("PART 1: Embedding Space Visualization")
    print("=" * 70)

    visualizer = EmbeddingVisualizer(
        model=model,
        test_dataset=test_ds,
        device=device,
        n_way=n_way,
        k_shot=k_shot,
        query_per_class=1,
        max_episode=args.max_episode,
        plots_dir=str(analysis_dir),
        reduction_method=args.reduction_method,
        jitter_strength=args.jitter_strength,
        random_state=42,
    )

    visualizer.visualize(verbose=True)

    # ================================================================
    # 2. HD vs 2D Comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: High-Dimensional vs 2D Prediction Comparison")
    print("=" * 70)

    all_results = []
    for ep_idx in range(args.max_episode):
        results = compare_hd_vs_2d_predictions(
            model=model,
            dataset=test_ds,
            episode_idx=ep_idx,
            query_sample_id=None,
            n_way=n_way,
            k_shot=k_shot,
            query_per_class=1,
            device=device,
            random_seed=42,
        )
        all_results.append(results)

    # ---- save combined comparison CSV ----
    csv_path = analysis_dir / "comparison.csv"
    rows = []
    for res in all_results:
        for sample in res["query_samples"]:
            rows.append({
                "episode": res["episode_idx"] + 1,
                "sample_id": sample["sample_id"],
                "true_label": sample["true_label"],
                "pred_hd": sample["pred_hd"],
                "pred_2d": sample["pred_2d"],
                "correct_hd": sample["correct_hd"],
                "mismatch": sample["mismatch"],
                "dist_BK_hd": sample["dists_hd"]["BK"],
                "dist_FK_hd": sample["dists_hd"]["FK"],
                "dist_BK_2d": sample["dists_2d"]["BK"],
                "dist_FK_2d": sample["dists_2d"]["FK"],
            })

    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nComparison results saved to: {csv_path}")

    # ---- summary ----
    total = len(rows)
    correct = sum(1 for r in rows if r["correct_hd"])
    mismatches = sum(1 for r in rows if r["mismatch"])

    print(f"\n{'#'*70}")
    print(f"  ANALYSIS {exp_id} COMPLETE")
    print(f"{'#'*70}")
    print(f"  Episodes analyzed: {args.max_episode}")
    print(f"  Total query samples: {total}")
    print(f"  HD accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  HD vs 2D mismatches: {mismatches}/{total} ({100*mismatches/total:.1f}%)")
    print(f"  Output: {analysis_dir}")
    print(f"  Plots:  episode_1.svg .. episode_{args.max_episode}.svg")
    print(f"  CSV:    comparison.csv")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
