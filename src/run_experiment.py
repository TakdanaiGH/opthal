#!/usr/bin/env python3
"""
run_experiment.py
=================
Experiment runner for the Cosine Prototypical Network.

Uses the professor's Trainer and Evaluator modules with label remapping
patches (from notebook Cell 7) to fix label-prototype misalignment.

Additionally fixes the evaluation metric bugs:
  - decode_label now uses GLOBAL labels (not local episode labels)
  - Confusion matrix uses GLOBAL labels consistently across episodes
  - ROC/AUC uses correct positive-class probability across episodes

Configurable via CLI (or via run_experiment.sh):
  --backbone                   resnet50 | vit | dinov2
  --embedding_dims             e.g. 512 256
  --open_final_layer_backbone  true | false
  --data_augmentation          true | false
  --episodes                   100 | 300  (training episodes per epoch)

Every run gets a unique experiment ID.  All artefacts are tagged with it:
  models/model_{id}.pth
  logs/log_{id}.json
  logs/loss_curve_{id}.png
  results/predictions_{id}.csv
  plots/roc_{id}.svg

A single meta_experiments.csv (project root) accumulates one row per run.
"""

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- project modules ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module.preprocess import ImageRegistryBuilder
from module.config import RuntimeConfig
from module.dataspliter import DataSplitter
from module.dataset import ImageDataset
from module.transform import ImageTransforms
from module.model import PrototypicalNetwork
from module.trainer import PrototypicalNetworkTrainer
from module.evaluator import ModelEvaluator


# ============================================================
# CLI
# ============================================================
def _parse_bool(v: str) -> bool:
    return v.strip().lower() in ("true", "t", "1", "yes")


def parse_args():
    p = argparse.ArgumentParser(description="Run a Prototypical Network experiment")
    p.add_argument("--backbone", type=str, default="resnet50",
                   choices=["resnet50", "vit", "dinov2"])
    p.add_argument("--embedding_dims", type=int, nargs="+", default=[512, 256],
                   help="Embedding layer dimensions, e.g. 512 256")
    p.add_argument("--open_final_layer_backbone", type=str, default="false",
                   help="Unfreeze final backbone block (true/false)")
    p.add_argument("--data_augmentation", type=str, default="true",
                   help="Enable training augmentation (true/false)")
    p.add_argument("--episodes", type=int, default=100,
                   help="Episodes per epoch for TRAINING (100 or 300)")
    return p.parse_args()


# ============================================================
# Label Remapping Fix (from professor's notebook Cell 7)
# ============================================================
# The prototypical network assigns local indices (0, 1, ...) to prototypes
# based on the ORDER classes appear in the support set. But the original
# Trainer/Evaluator compared these local indices against the dataset's
# GLOBAL label encoding. When classes are sampled in reversed order,
# ~50% of labels are wrong -> training signal cancels out.
#
# Fix: remap labels from global -> local episode indices before loss.
# ============================================================
def _remap_labels(labels, n_way, k_shot):
    """Remap global dataset labels to local episode indices (0 .. n_way-1)."""
    support_size = n_way * k_shot
    support_labels = labels[:support_size]
    local_map = {}
    for way_idx in range(n_way):
        global_label = support_labels[way_idx * k_shot].item()
        local_map[global_label] = way_idx
    return torch.tensor([local_map[l.item()] for l in labels],
                        dtype=labels.dtype, device=labels.device)


# ============================================================
# Patched Trainer methods (from professor's notebook Cell 7)
# ============================================================
def _patched_train_epoch(self) -> float:
    self.model.train()
    epoch_loss = 0.0
    train_loader = DataLoader(
        self.train_dataset, batch_sampler=self.train_sampler,
        collate_fn=self.collate_fn)
    for images, labels, _, _ in train_loader:
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = _remap_labels(labels, self.n_way, self.k_shot)
        support_size = self.n_way * self.k_shot
        self.optimizer.zero_grad()
        logits, _ = self.model(images[:support_size], images[support_size:],
                               self.n_way, self.k_shot)
        loss = self.criterion(logits, labels[support_size:])
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


def _patched_validate_epoch(self) -> float:
    self.model.eval()
    epoch_loss = 0.0
    val_loader = DataLoader(
        self.val_dataset, batch_sampler=self.val_sampler,
        collate_fn=self.collate_fn)
    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = _remap_labels(labels, self.n_way, self.k_shot)
            support_size = self.n_way * self.k_shot
            logits, _ = self.model(images[:support_size], images[support_size:],
                                   self.n_way, self.k_shot)
            loss = self.criterion(logits, labels[support_size:])
            epoch_loss += loss.item()
    return epoch_loss / len(val_loader)


# ============================================================
# Patched Evaluator method
# ============================================================
# Beyond the professor's Cell 7 patch, this also fixes:
#   1. decode_label: uses GLOBAL encoded label (not local episode label)
#   2. Confusion matrix: uses GLOBAL labels so TP/FP/TN/FN are consistent
#      across episodes (professor's code randomly swaps positive/negative)
#   3. ROC/AUC: always uses probability of GLOBAL class 1 as positive score
# ============================================================
def _patched_evaluate(self, verbose=True):
    print(f"\n{'='*70}")
    print("Starting Model Evaluation on Test Data")
    print(f"{'='*70}")
    print(f"  Test Episodes: {self.max_episode}")
    print(f"  {self.n_way}-way {self.k_shot}-shot, query={self.query_per_class}")
    print(f"  Device: {self.device}")
    print(f"{'='*70}\n")

    self.model.eval()
    test_loader = DataLoader(
        self.test_dataset, batch_sampler=self.test_sampler,
        collate_fn=self.collate_fn)

    with torch.no_grad():
        for ep_idx, (images, labels, original_labels, paths) in enumerate(test_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Keep original global encoded labels BEFORE remapping
            global_labels = labels.clone()
            remapped = _remap_labels(labels, self.n_way, self.k_shot)

            sup = self.n_way * self.k_shot

            # Build local -> global mapping for this episode
            local_to_global = {}
            for way_idx in range(self.n_way):
                global_enc = global_labels[way_idx * self.k_shot].item()
                local_to_global[way_idx] = global_enc
            global_to_local = {v: k for k, v in local_to_global.items()}

            logits, _ = self.model(images[:sup], images[sup:],
                                   self.n_way, self.k_shot)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            pred_probs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            q_preds = preds.cpu().numpy()
            q_probs = pred_probs.cpu().numpy()
            q_all_probs = probs.cpu().numpy()
            q_orig = original_labels[sup:]
            q_paths = paths[sup:]

            for i in range(len(q_preds)):
                pred_local = int(q_preds[i])

                # Convert to GLOBAL encoded labels for consistent metrics
                true_global = int(global_labels[sup + i].item())
                pred_global = int(local_to_global[pred_local])

                # Confusion matrix with global labels
                self._update_confusion_matrix(true_global, pred_global)
                error_type = self._get_error_type(true_global, pred_global)

                # Decode predicted label using GLOBAL encoded label (correct)
                predicted_label_str = self.test_dataset.decode_label(pred_global)

                self.predictions.append({
                    'episode_id': ep_idx + 1,
                    'file_name': os.path.basename(q_paths[i]),
                    'prediction_score': float(q_probs[i]),
                    'class_label': q_orig[i],
                    'predicted_label': predicted_label_str,
                    'error_type': error_type,
                })

                # ROC curve with global labels
                self.true_labels.append(true_global)
                self.pred_labels.append(pred_global)
                if self.n_way == 2 and 1 in global_to_local:
                    # Always use probability of GLOBAL class 1 as positive score
                    local_idx_for_positive = global_to_local[1]
                    self.pred_scores.append(float(q_all_probs[i, local_idx_for_positive]))
                else:
                    self.pred_scores.append(float(q_probs[i]))

            if verbose and (ep_idx + 1) % 10 == 0:
                print(f"  {ep_idx+1}/{self.max_episode} episodes done")

    print(f"\n  Completed evaluation on {self.max_episode} episodes")
    print(f"  Total query samples: {len(self.predictions)}\n")

    metrics = self._compute_metrics()
    self._save_predictions()
    if self.n_way == 2:
        self._plot_roc_curve()
    self._print_metrics_report(metrics)
    return metrics


# ============================================================
# Patched save methods (use exp_id instead of backbone_name)
# ============================================================
# The professor's modules save as {backbone_name}.pth etc.
# We patch them to use _exp_id (set on each instance in main).
# ============================================================
def _patched_save_checkpoint(self, epoch):
    tag = getattr(self, '_exp_id', self.model.backbone_name)
    checkpoint_path = self.checkpoint_dir / f"model_{tag}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'val_loss': self.best_val_loss,
        'n_way': self.n_way,
        'k_shot': self.k_shot,
        'query_per_class_train': self.query_per_class_train,
        'query_per_class_val': self.query_per_class_val,
    }, checkpoint_path)
    print(f"  Model checkpoint saved to: {checkpoint_path}")


def _patched_save_log(self):
    tag = getattr(self, '_exp_id', self.model.backbone_name)
    log_path = self.log_dir / f"log_{tag}.json"
    with open(log_path, 'w') as f:
        json.dump(self.history, f, indent=4)
    print(f"  Training log saved to: {log_path}")


def _patched_plot_training_curves(self):
    tag = getattr(self, '_exp_id', self.model.backbone_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(self.history['epoch'], self.history['train_loss'],
            'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax.plot(self.history['epoch'], self.history['val_loss'],
            'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax.axvline(x=self.best_epoch, color='g', linestyle='--',
               label=f'Best Epoch ({self.best_epoch})', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training and Validation Loss [{tag}]', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = self.log_dir / f"loss_curve_{tag}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Loss curve saved to: {plot_path}")


def _patched_save_predictions(self):
    tag = getattr(self, '_exp_id', self.model.backbone_name)
    csv_path = self.results_dir / f"predictions_{tag}.csv"
    pd.DataFrame(self.predictions).to_csv(csv_path, index=False)
    print(f"  Predictions saved to: {csv_path}")


def _patched_plot_roc_curve(self):
    from sklearn.metrics import roc_curve, auc
    tag = getattr(self, '_exp_id', self.model.backbone_name)
    fpr, tpr, _ = roc_curve(self.true_labels, self.pred_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve [{tag}]', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = self.plots_dir / f"roc_{tag}.svg"
    plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ROC curve saved to: {plot_path}")
    return roc_auc


# ============================================================
# Apply all patches
# ============================================================
PrototypicalNetworkTrainer._train_epoch = _patched_train_epoch
PrototypicalNetworkTrainer._validate_epoch = _patched_validate_epoch
PrototypicalNetworkTrainer._save_checkpoint = _patched_save_checkpoint
PrototypicalNetworkTrainer._save_log = _patched_save_log
PrototypicalNetworkTrainer._plot_training_curves = _patched_plot_training_curves
ModelEvaluator.evaluate = _patched_evaluate
ModelEvaluator._save_predictions = _patched_save_predictions
ModelEvaluator._plot_roc_curve = _patched_plot_roc_curve


# ============================================================
# Helpers
# ============================================================
def unfreeze_final_layers(model):
    """Selectively unfreeze only the last block of the frozen backbone."""
    name = model.backbone_name
    if name == "resnet50":
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
    elif name in ("vit", "vit_b_16"):
        for p in model.backbone.encoder.layers[-1].parameters():
            p.requires_grad = True
        for p in model.backbone.encoder.ln.parameters():
            p.requires_grad = True
    elif name in ("dinov2", "dinov2_vits14"):
        for p in model.backbone.blocks[-1].parameters():
            p.requires_grad = True
        for p in model.backbone.norm.parameters():
            p.requires_grad = True


# ============================================================
# Constants
# ============================================================
N_WAY = 2
K_SHOT = 3
QUERY_TRAIN = 3
QUERY_VAL = 2
MAX_EPOCH = 100
LR = 0.001
PATIENCE = 20

K_SHOT_EVAL = 3
QUERY_EVAL = 1
EVAL_EPISODES = 100


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    open_final = _parse_bool(args.open_final_layer_backbone)
    augment    = _parse_bool(args.data_augmentation)
    emb_dims   = tuple(args.embedding_dims)
    exp_id     = uuid.uuid4().hex[:8]

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT  {exp_id}")
    print(f"{'#'*70}")
    print(f"  backbone               = {args.backbone}")
    print(f"  embedding_dims         = {emb_dims}")
    print(f"  open_final_layer       = {open_final}")
    print(f"  data_augmentation      = {augment}")
    print(f"  episodes (train)       = {args.episodes}")
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
    tx = ImageTransforms(image_size=224, augment_training=augment)

    # ---- datasets ----
    train_ds = ImageDataset(train_df, root_dir="..",
                            transform=tx.get_train_transform())
    val_ds   = ImageDataset(val_df,   root_dir="..",
                            transform=tx.get_test_transform(),
                            label_encoder=train_ds.get_label_encoder())
    test_ds  = ImageDataset(test_df,  root_dir="..",
                            transform=tx.get_test_transform(),
                            label_encoder=train_ds.get_label_encoder())
    print(f"Datasets  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ---- model ----
    model = PrototypicalNetwork(
        backbone=args.backbone,
        embedding_dims=emb_dims,
        pretrained=True,
        cache_dir="../backbones",
    )
    model.freeze_backbone()
    if open_final:
        unfreeze_final_layers(model)
    model = model.to(device)
    model.print_model_info()

    # ---- train (professor's Trainer with remap patches) ----
    trainer = PrototypicalNetworkTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        n_way=N_WAY,
        k_shot=K_SHOT,
        query_per_class_train=QUERY_TRAIN,
        query_per_class_val=QUERY_VAL,
        max_epoch=MAX_EPOCH,
        max_episode=args.episodes,
        learning_rate=LR,
        patience=PATIENCE,
        checkpoint_dir='../models',
        log_dir='../logs',
    )
    trainer._exp_id = exp_id
    history = trainer.train(verbose=True)
    best_val_loss = trainer.best_val_loss
    best_epoch = trainer.best_epoch

    # Reload best checkpoint
    ckpt_path = trainer.checkpoint_dir / f"model_{exp_id}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # ---- evaluate (professor's Evaluator with remap + global-label fixes) ----
    evaluator = ModelEvaluator(
        model=model,
        test_dataset=test_ds,
        device=device,
        n_way=N_WAY,
        k_shot=K_SHOT_EVAL,
        query_per_class=QUERY_EVAL,
        max_episode=EVAL_EPISODES,
        results_dir='../results',
        plots_dir='../plots',
    )
    evaluator._exp_id = exp_id
    metrics = evaluator.evaluate(verbose=True)

    # ---- compute AUC from evaluator data ----
    auc_val = None
    if N_WAY == 2 and evaluator.pred_scores:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(evaluator.true_labels, evaluator.pred_scores)
        auc_val = auc(fpr, tpr)

    # ---- append to meta_experiments.csv ----
    accuracy  = metrics['accuracy']['value']
    precision = metrics['precision']['value']
    recall    = metrics['recall']['value']
    f1_score  = metrics['f1_score']['value']

    meta_path = Path("../meta_experiments.csv")
    file_exists = meta_path.exists()

    row = {
        "open_final_layer_backbone": open_final,
        "backbone": args.backbone,
        "embedding_dims": str(list(emb_dims)),
        "data_augmentation": augment,
        "episodes": args.episodes,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "auc": round(auc_val, 4) if auc_val is not None else "",
        "experiment_id": exp_id,
    }

    with open(meta_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT {exp_id} COMPLETE")
    print(f"  Accuracy={accuracy:.3f}  F1={f1_score:.3f}", end="")
    if auc_val is not None:
        print(f"  AUC={auc_val:.3f}", end="")
    print()
    print(f"  Meta CSV: {meta_path}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
