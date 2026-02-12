from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from pathlib import Path
from module.model import PrototypicalNetwork
from module.episode import FewShotBatchSampler
from module.dataset import ImageDataset

from sklearn.metrics import roc_curve, auc
import scipy.stats as stats


class ModelEvaluator:
    """
    Evaluator for Prototypical Network on test data.
    
    Performs episode-based testing, computes classification metrics with confidence
    intervals, generates ROC curves, and saves detailed predictions.
    """
    
    def __init__(self, model: PrototypicalNetwork, test_dataset: ImageDataset,
                 device: torch.device, n_way: int = 2, k_shot: int = 5,
                 query_per_class: int = 5, max_episode: int = 100,
                 results_dir: str = '../results', plots_dir: str = '../plots'):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained PrototypicalNetwork instance
            test_dataset: Test dataset
            device: Device to run evaluation on
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            query_per_class: Number of query examples per class
            max_episode: Number of test episodes
            results_dir: Directory to save prediction results
            plots_dir: Directory to save plots
        """
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.max_episode = max_episode
        
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.predictions = []
        self.true_labels = []
        self.pred_labels = []
        self.pred_scores = []
        self.file_names = []
        self.episode_ids = []
        
        # Confusion matrix
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        
        # Create test sampler
        self.test_sampler = FewShotBatchSampler(
            dataset=self.test_dataset,
            n_way=self.n_way,
            k_shot=self.k_shot,
            query_per_class=self.query_per_class,
            n_episodes=self.max_episode,
            episodes_per_batch=1
        )
        
        # Custom collate function
        self.collate_fn = lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch]),
            [item[2] for item in batch],
            [item[3] for item in batch]
        )
    
    def _update_confusion_matrix(self, true_label: int, pred_label: int):
        """
        Update confusion matrix based on prediction.
        
        Args:
            true_label: True label (0 or 1)
            pred_label: Predicted label (0 or 1)
        """
        if true_label == 1 and pred_label == 1:
            self.tp += 1
        elif true_label == 0 and pred_label == 1:
            self.fp += 1
        elif true_label == 0 and pred_label == 0:
            self.tn += 1
        elif true_label == 1 and pred_label == 0:
            self.fn += 1
    
    def _get_error_type(self, true_label: int, pred_label: int) -> str:
        """
        Get error type string.
        
        Args:
            true_label: True label (0 or 1)
            pred_label: Predicted label (0 or 1)
            
        Returns:
            Error type: 'TP', 'FP', 'TN', or 'FN'
        """
        if true_label == 1 and pred_label == 1:
            return 'TP'
        elif true_label == 0 and pred_label == 1:
            return 'FP'
        elif true_label == 0 and pred_label == 0:
            return 'TN'
        else:  # true_label == 1 and pred_label == 0
            return 'FN'
    
    def _compute_confidence_interval(self, proportion: float, n: int, 
                                     confidence: float = 0.95) -> tuple:
        """
        Compute Wilson score confidence interval for a proportion.
        
        Args:
            proportion: Observed proportion (e.g., accuracy)
            n: Sample size
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound), both trimmed to [0, 1]
        """
        if n == 0:
            return (0.0, 0.0)
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        # Wilson score interval
        center = (proportion + z**2 / (2 * n)) / (1 + z**2 / n)
        margin = z * np.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        
        lower = center - margin
        upper = center + margin
        
        # Trim to [0, 1]
        lower = max(0.0, lower)
        upper = min(1.0, upper)
        
        return (lower, upper)
    
    def evaluate(self, verbose: bool = True) -> dict:
        """
        Evaluate model on test data.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'='*70}")
        print("Starting Model Evaluation on Test Data")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Test Episodes: {self.max_episode}")
        print(f"  N-way: {self.n_way}, K-shot: {self.k_shot}, Query per class: {self.query_per_class}")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")
        
        self.model.eval()
        
        # Create test loader
        test_loader = DataLoader(
            self.test_dataset,
            batch_sampler=self.test_sampler,
            collate_fn=self.collate_fn
        )
        
        with torch.no_grad():
            for episode_idx, (images, labels, original_labels, paths) in enumerate(test_loader):
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Split into support and query
                support_size = self.n_way * self.k_shot
                support_images = images[:support_size]
                query_images = images[support_size:]
                query_labels = labels[support_size:]
                query_original_labels = original_labels[support_size:]
                query_paths = paths[support_size:]
                
                # Forward pass
                logits, _ = self.model(support_images, query_images,
                                       self.n_way, self.k_shot)
                
                # Get predictions and probabilities
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Get prediction scores (probability of predicted class)
                pred_probs = probabilities.gather(1, predictions.unsqueeze(1)).squeeze(1)
                
                # Move to CPU for processing
                query_labels_cpu = query_labels.cpu().numpy()
                predictions_cpu = predictions.cpu().numpy()
                pred_probs_cpu = pred_probs.cpu().numpy()
                probabilities_cpu = probabilities.cpu().numpy()
                
                # Store predictions for each query sample
                for i in range(len(query_labels_cpu)):
                    true_label = query_labels_cpu[i]
                    pred_label = predictions_cpu[i]
                    pred_score = pred_probs_cpu[i]
                    file_name = os.path.basename(query_paths[i])
                    
                    # Update confusion matrix
                    self._update_confusion_matrix(true_label, pred_label)
                    
                    # Get error type
                    error_type = self._get_error_type(true_label, pred_label)
                    
                    # Store prediction details
                    self.predictions.append({
                        'episode_id': episode_idx + 1,
                        'file_name': file_name,
                        'prediction_score': pred_score,
                        'class_label': query_original_labels[i],
                        'predicted_label': self.test_dataset.decode_label(pred_label),
                        'error_type': error_type
                    })
                    
                    # Store for ROC curve (use probability of positive class)
                    self.true_labels.append(true_label)
                    self.pred_labels.append(pred_label)
                    # For binary classification, use probability of class 1
                    if self.n_way == 2:
                        self.pred_scores.append(probabilities_cpu[i, 1])
                    else:
                        self.pred_scores.append(pred_score)
                
                if verbose and (episode_idx + 1) % 10 == 0:
                    print(f"Processed {episode_idx + 1}/{self.max_episode} episodes...")
        
        print(f"\n✓ Completed evaluation on {self.max_episode} episodes")
        print(f"✓ Total query samples evaluated: {len(self.predictions)}\n")
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Save predictions
        self._save_predictions()
        
        # Plot ROC curve
        if self.n_way == 2:
            self._plot_roc_curve()
        
        # Print metrics report
        self._print_metrics_report(metrics)
        
        return metrics
    
    def _compute_metrics(self) -> dict:
        """
        Compute classification metrics with confidence intervals.
        
        Returns:
            Dictionary containing all metrics
        """
        total = self.tp + self.fp + self.tn + self.fn
        
        # Compute metrics
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        accuracy = (self.tp + self.tn) / total if total > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute confidence intervals
        precision_ci = self._compute_confidence_interval(precision, self.tp + self.fp)
        recall_ci = self._compute_confidence_interval(recall, self.tp + self.fn)
        accuracy_ci = self._compute_confidence_interval(accuracy, total)
        f1_ci = self._compute_confidence_interval(f1, total)
        
        return {
            'confusion_matrix': {
                'TP': self.tp,
                'FP': self.fp,
                'TN': self.tn,
                'FN': self.fn,
                'total': total
            },
            'precision': {
                'value': precision,
                'ci_lower': precision_ci[0],
                'ci_upper': precision_ci[1]
            },
            'recall': {
                'value': recall,
                'ci_lower': recall_ci[0],
                'ci_upper': recall_ci[1]
            },
            'accuracy': {
                'value': accuracy,
                'ci_lower': accuracy_ci[0],
                'ci_upper': accuracy_ci[1]
            },
            'f1_score': {
                'value': f1,
                'ci_lower': f1_ci[0],
                'ci_upper': f1_ci[1]
            }
        }
    
    def _save_predictions(self):
        """Save predictions to CSV file."""
        predictions_df = pd.DataFrame(self.predictions)
        csv_path = self.results_dir / f"{self.model.backbone_name}_predictions.csv"
        predictions_df.to_csv(csv_path, index=False)
        print(f"✓ Predictions saved to: {csv_path}")
    
    def _plot_roc_curve(self):
        """Plot and save ROC curve for binary classification."""
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(self.true_labels, self.pred_scores)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal reference line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{self.model.backbone_name}_roc_curve.svg"
        plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {plot_path}")
        
        plt.show()
        
        return roc_auc
    
    def _print_metrics_report(self, metrics: dict):
        """Print formatted metrics report."""
        print(f"\n{'='*70}")
        print("Evaluation Metrics Report")
        print(f"{'='*70}\n")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        print("Confusion Matrix:")
        print(f"  TP (True Positive):  {cm['TP']:4d}")
        print(f"  FP (False Positive): {cm['FP']:4d}")
        print(f"  TN (True Negative):  {cm['TN']:4d}")
        print(f"  FN (False Negative): {cm['FN']:4d}")
        print(f"  Total Samples:       {cm['total']:4d}\n")
        
        # Classification metrics with 95% CI
        print("Classification Metrics (95% Confidence Interval):")
        print("-" * 70)
        
        for metric_name in ['precision', 'recall', 'accuracy', 'f1_score']:
            metric = metrics[metric_name]
            name = metric_name.replace('_', ' ').title()
            print(f"  {name:12s}: {metric['value']:.3f} "
                  f"({metric['ci_lower']:.3f}, {metric['ci_upper']:.3f})")
        
        print(f"\n{'='*70}\n")