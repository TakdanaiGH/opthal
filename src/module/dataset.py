from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Tuple, Dict

class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images with encoded labels.
    """
    
    def __init__(self, dataframe: pd.DataFrame, root_dir: str = None, 
                 transform=None, label_encoder: dict = None):
        """
        Initialize the ImageDataset.
        
        Args:
            dataframe: DataFrame with 'image_path' and 'class_label' columns
            root_dir: Optional root directory to prepend to image paths
            transform: Optional transform to apply to images
            label_encoder: Optional dictionary mapping labels to integers
                          If None, will be created automatically
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        
        # Create label encoder if not provided
        if label_encoder is None:
            unique_labels = sorted(self.dataframe['class_label'].unique())
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_encoder = label_encoder
        
        # Create reverse mapping (decoder)
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        
        # Encode labels
        self.dataframe['encoded_label'] = self.dataframe['class_label'].map(self.label_encoder)
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label, original_label, image_path)
        """
        # Get image path and label
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        
        # Prepend root_dir if provided
        if self.root_dir is not None:
            img_path = os.path.join(self.root_dir, img_path)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Get labels
        label = row['encoded_label']
        original_label = row['class_label']
        
        return image, label, original_label, img_path
    
    def get_label_encoder(self) -> dict:
        """Return the label encoder dictionary."""
        return self.label_encoder
    
    def get_label_decoder(self) -> dict:
        """Return the label decoder dictionary."""
        return self.label_decoder
    
    def decode_label(self, encoded_label: int) -> str:
        """
        Decode an encoded label back to original string.
        
        Args:
            encoded_label: Encoded integer label
            
        Returns:
            Original string label
        """
        return self.label_decoder[encoded_label]
    
    def get_class_counts(self) -> dict:
        """Return the count of samples per class."""
        return dict(Counter(self.dataframe['class_label']))
    
    def get_samples_by_class(self, class_label: str) -> pd.DataFrame:
        """
        Get all samples belonging to a specific class.
        
        Args:
            class_label: The class label to filter by
            
        Returns:
            DataFrame containing only samples from the specified class
        """
        return self.dataframe[self.dataframe['class_label'] == class_label]
    
    def print_info(self) -> None:
        """Print information about the dataset."""
        print(f"\n{'='*60}")
        print("Dataset Information")
        print(f"{'='*60}")
        print(f"Total samples: {len(self)}")
        print(f"Number of classes: {len(self.label_encoder)}")
        print(f"\nLabel Encoding:")
        for label, idx in self.label_encoder.items():
            count = len(self.get_samples_by_class(label))
            print(f"  {label} → {idx} ({count} samples)")
        print(f"{'='*60}\n")