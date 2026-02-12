import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Tuple, Dict



class DataSplitter:
    """
    A class to split dataset into train, validation, and test sets with stratification.
    """
    
    def __init__(self, csv_path: str, train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, test_ratio: float = 0.15, 
                 random_state: int = 42):
        """
        Initialize the DataSplitter.
        
        Args:
            csv_path: Path to the CSV file containing image registry
            train_ratio: Proportion of data for training (default: 0.7)
            val_ratio: Proportion of data for validation (default: 0.15)
            test_ratio: Proportion of data for testing (default: 0.15)
            random_state: Random seed for reproducibility
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        self.csv_path = csv_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV file into a DataFrame."""
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(self.df)} samples from {self.csv_path}")
        return self.df
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train, validation, and test sets with stratification.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            self.load_data()
        
        # First split: separate out test set
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=self.test_ratio,
            stratify=self.df['class_label'],
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        # Adjust val_ratio relative to remaining data
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df['class_label'],
            random_state=self.random_state
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print(f"\n{'='*60}")
        print("Data Split Summary")
        print(f"{'='*60}")
        print(f"Train set: {len(train_df)} samples ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"Val set:   {len(val_df)} samples ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"Test set:  {len(test_df)} samples ({len(test_df)/len(self.df)*100:.1f}%)")
        print(f"Total:     {len(self.df)} samples")
        print(f"{'='*60}\n")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get class distribution as percentages.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping class labels to percentages
        """
        counts = Counter(df['class_label'])
        total = len(df)
        return {label: (count / total) * 100 for label, count in counts.items()}
    
    def plot_distributions(self, figsize: Tuple[int, int] = (15, 4)) -> None:
        """
        Plot class distributions for train, val, and test sets.
        
        Args:
            figsize: Figure size as (width, height)
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        
        # Get class distributions
        train_dist = self.get_class_distribution(self.train_df)
        val_dist = self.get_class_distribution(self.val_df)
        test_dist = self.get_class_distribution(self.test_df)
        
        # Get all unique class labels
        all_classes = sorted(set(self.df['class_label'].unique()))
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot train distribution
        train_percentages = [train_dist.get(cls, 0) for cls in all_classes]
        axes[0].bar(all_classes, train_percentages, color='steelblue', alpha=0.8)
        axes[0].set_title('Train Set Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Class Label', fontsize=10)
        axes[0].set_ylabel('Percentage (%)', fontsize=10)
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(train_percentages):
            axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot validation distribution
        val_percentages = [val_dist.get(cls, 0) for cls in all_classes]
        axes[1].bar(all_classes, val_percentages, color='forestgreen', alpha=0.8)
        axes[1].set_title('Validation Set Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Class Label', fontsize=10)
        axes[1].set_ylabel('Percentage (%)', fontsize=10)
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(val_percentages):
            axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot test distribution
        test_percentages = [test_dist.get(cls, 0) for cls in all_classes]
        axes[2].bar(all_classes, test_percentages, color='coral', alpha=0.8)
        axes[2].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Class Label', fontsize=10)
        axes[2].set_ylabel('Percentage (%)', fontsize=10)
        axes[2].set_ylim(0, 100)
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(test_percentages):
            axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig("../plots/class_distributions.svg", format="svg")
        plt.show()
        
        print("Class Distribution Summary:")
        print(f"{'Split':<12} {'Class':<10} {'Percentage':<12} {'Count':<8}")
        print("-" * 50)
        for split_name, split_df in [('Train', self.train_df), 
                                       ('Validation', self.val_df), 
                                       ('Test', self.test_df)]:
            dist = self.get_class_distribution(split_df)
            counts = Counter(split_df['class_label'])
            for cls in all_classes:
                print(f"{split_name:<12} {cls:<10} {dist.get(cls, 0):>6.1f}%      {counts.get(cls, 0):>5}")
    
    def save_splits(self, train_path: str = '../train.csv', 
                    val_path: str = '../val.csv', 
                    test_path: str = '../test.csv') -> None:
        """
        Save the split datasets to separate CSV files.
        
        Args:
            train_path: Path to save training set
            val_path: Path to save validation set
            test_path: Path to save test set
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Data not split yet. Call split_data() first.")
        
        self.train_df.to_csv(train_path, index=False)
        self.val_df.to_csv(val_path, index=False)
        self.test_df.to_csv(test_path, index=False)
        
        print(f"✓ Train set saved to: {train_path}")
        print(f"✓ Validation set saved to: {val_path}")
        print(f"✓ Test set saved to: {test_path}")


