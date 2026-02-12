import json
import time
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from module.model import PrototypicalNetwork
from module.episode import FewShotBatchSampler
from module.dataset import ImageDataset

class PrototypicalNetworkTrainer:
    """
    Trainer for Prototypical Network using episode-based learning.
    
    Implements training loop with validation, early stopping, model checkpointing,
    and automatic logging of training metrics.
    """
    
    def __init__(self, model: PrototypicalNetwork, train_dataset: ImageDataset,
                 val_dataset: ImageDataset, device: torch.device,
                 n_way: int = 2, k_shot: int = 5, 
                 query_per_class_train: int = 5, query_per_class_val: int = None,
                 max_epoch: int = 100, max_episode: int = 100,
                 learning_rate: float = 0.001, patience: int = 10,
                 checkpoint_dir: str = '../models', log_dir: str = '../logs'):
        """
        Initialize the trainer.
        
        Args:
            model: PrototypicalNetwork instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: Device to run training on
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            query_per_class_train: Number of query examples per class for training
            query_per_class_val: Number of query examples per class for validation
                                 (defaults to query_per_class_train if None)
            max_epoch: Maximum number of epochs
            max_episode: Number of episodes per epoch
            learning_rate: Learning rate for optimizer
            patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class_train = query_per_class_train
        self.query_per_class_val = query_per_class_val if query_per_class_val is not None else query_per_class_train
        
        self.max_epoch = max_epoch
        self.max_episode = max_episode
        self.learning_rate = learning_rate
        self.patience = patience
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'timestamp': []
        }
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
        # Create episode samplers
        self._create_samplers()
        
        # Custom collate function
        self.collate_fn = lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch]),
            [item[2] for item in batch],
            [item[3] for item in batch]
        )
    
    def _create_samplers(self):
        """Create episode samplers for training and validation."""
        self.train_sampler = FewShotBatchSampler(
            dataset=self.train_dataset,
            n_way=self.n_way,
            k_shot=self.k_shot,
            query_per_class=self.query_per_class_train,
            n_episodes=self.max_episode,
            episodes_per_batch=1
        )
        
        self.val_sampler = FewShotBatchSampler(
            dataset=self.val_dataset,
            n_way=self.n_way,
            k_shot=self.k_shot,
            query_per_class=self.query_per_class_val,
            n_episodes=self.max_episode,
            episodes_per_batch=1
        )
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn
        )
        
        for episode_idx, (images, labels, _, _) in enumerate(train_loader):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Split into support and query
            support_size = self.n_way * self.k_shot
            support_images = images[:support_size]
            query_images = images[support_size:]
            query_labels = labels[support_size:]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(support_images, query_images, 
                                   self.n_way, self.k_shot)
            
            # Compute loss
            loss = self.criterion(logits, query_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
        
        # Return average loss
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    def _validate_epoch(self) -> float:
        """
        Validate for one epoch.
        
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        epoch_loss = 0.0
        
        # Create data loader
        val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            collate_fn=self.collate_fn
        )
        
        with torch.no_grad():
            for episode_idx, (images, labels, _, _) in enumerate(val_loader):
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Split into support and query
                support_size = self.n_way * self.k_shot
                support_images = images[:support_size]
                query_images = images[support_size:]
                query_labels = labels[support_size:]
                
                # Forward pass
                logits, _ = self.model(support_images, query_images,
                                       self.n_way, self.k_shot)
                
                # Compute loss
                loss = self.criterion(logits, query_labels)
                
                # Accumulate loss
                epoch_loss += loss.item()
        
        # Return average loss
        avg_loss = epoch_loss / len(val_loader)
        return avg_loss
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{self.model.backbone_name}.pth"
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
        print(f"✓ Model checkpoint saved to: {checkpoint_path}")
    
    def _save_log(self):
        """Save training log to JSON file."""
        log_path = self.log_dir / f"{self.model.backbone_name}_training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"✓ Training log saved to: {log_path}")
    
    def _plot_training_curves(self):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = self.history['epoch']
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']
        
        ax.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        
        # Mark best epoch
        ax.axvline(x=self.best_epoch, color='g', linestyle='--', 
                   label=f'Best Epoch ({self.best_epoch})', alpha=0.7)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.log_dir / f"{self.model.backbone_name}_loss_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss curve saved to: {plot_path}")
        
        plt.show()
    
    def train(self, verbose: bool = True):
        """
        Train the model with early stopping.
        
        Args:
            verbose: Whether to print training progress
        """
        print(f"\n{'='*70}")
        print("Starting Episode-Based Training")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Max Epochs: {self.max_epoch}")
        print(f"  Episodes per Epoch: {self.max_episode}")
        print(f"  N-way: {self.n_way}, K-shot: {self.k_shot}")
        print(f"  Query per class (train): {self.query_per_class_train}")
        print(f"  Query per class (val): {self.query_per_class_val}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Early Stopping Patience: {self.patience}")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.max_epoch + 1):
            epoch_start_time = time.time()
            
            # Recreate samplers for new random episodes each epoch
            self._create_samplers()
            
            # Train one epoch
            train_loss = self._train_epoch()
            
            # Validate one epoch
            val_loss = self._validate_epoch()
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.learning_rate)
            self.history['timestamp'].append(datetime.now().isoformat())
            
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            if verbose:
                print(f"Epoch [{epoch:3d}/{self.max_epoch}] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Time: {epoch_time:.2f}s", end='')
            
            # Check if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch)
                if verbose:
                    print(" ✓ [Best Model]")
            else:
                self.epochs_without_improvement += 1
                if verbose:
                    print(f" (No improvement: {self.epochs_without_improvement}/{self.patience})")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered after {epoch} epochs")
                print(f"No improvement in validation loss for {self.patience} consecutive epochs")
                print(f"{'='*70}\n")
                break
        
        total_time = time.time() - start_time
        
        # Save training log
        self._save_log()
        
        # Print training summary
        print(f"\n{'='*70}")
        print("Training Summary")
        print(f"{'='*70}")
        print(f"Total Epochs: {len(self.history['epoch'])}")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Final Training Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final Validation Loss: {self.history['val_loss'][-1]:.4f}")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"{'='*70}\n")
        
        # Plot training curves
        self._plot_training_curves()
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path: str = None):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads from default location.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"{self.model.backbone_name}.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"✓ Validation loss: {checkpoint['val_loss']:.4f}")