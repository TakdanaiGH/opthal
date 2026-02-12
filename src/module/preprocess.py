import os
import csv
from pathlib import Path
from typing import List, Tuple

class ImageRegistryBuilder:
    """
    A class to build an image registry from a directory structure.
    
    The registry maps image file paths to their corresponding class labels,
    where class labels are derived from subfolder names.
    """
    
    def __init__(self, root_dir: str, image_extensions: List[str] = None):
        """
        Initialize the ImageRegistryBuilder.
        
        Args:
            root_dir: Root directory containing class subfolders
            image_extensions: List of valid image file extensions (default: common formats)
        """
        self.root_dir = Path(root_dir)
        if image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        else:
            self.image_extensions = image_extensions
        
        self.registry_data: List[Tuple[str, str]] = []
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if a file is an image based on its extension."""
        return any(filename.lower().endswith(ext) for ext in self.image_extensions)
    
    def scan_directory(self) -> None:
        """
        Scan the root directory and collect image paths with their labels.
        
        The directory structure should be:
            root_dir/
                class1/
                    image1.jpg
                    image2.jpg
                class2/
                    image3.jpg
        """
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        
        self.registry_data.clear()
        
        # Iterate through subdirectories (each represents a class)
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_label = class_dir.name
            
            # Iterate through files in each class directory
            for image_file in sorted(class_dir.iterdir()):
                if image_file.is_file() and self._is_image_file(image_file.name):
                    image_path = str(image_file)
                    self.registry_data.append((image_path, class_label))
    
    def save_to_csv(self, output_path: str, use_relative_paths: bool = True) -> None:
        """
        Save the registry data to a CSV file.
        
        Args:
            output_path: Path where the CSV file will be saved
            use_relative_paths: If True, store paths relative to root_dir
        """
        if not self.registry_data:
            raise ValueError("No data to save. Run scan_directory() first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'class_label'])
            
            for image_path, class_label in self.registry_data:
                if use_relative_paths:
                    # Make path relative to the parent of root_dir
                    image_path = str(Path(image_path).relative_to(self.root_dir.parent))
                writer.writerow([image_path, class_label])
        
        print(f"✓ Registry saved to: {output_file}")
        print(f"✓ Total images: {len(self.registry_data)}")
        print(f"✓ Total classes: {len(set(label for _, label in self.registry_data))}")
    
    def get_summary(self) -> dict:
        """
        Get a summary of the registry data.
        
        Returns:
            Dictionary containing summary statistics
        """
        class_counts = {}
        for _, class_label in self.registry_data:
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        return {
            'total_images': len(self.registry_data),
            'total_classes': len(class_counts),
            'class_distribution': class_counts
        }
    
    def print_summary(self) -> None:
        """Print a summary of the registry data."""
        summary = self.get_summary()
        print(f"\n{'='*50}")
        print("Image Registry Summary")
        print(f"{'='*50}")
        print(f"Total Images: {summary['total_images']}")
        print(f"Total Classes: {summary['total_classes']}")
        print(f"\nClass Distribution:")
        for class_name, count in sorted(summary['class_distribution'].items()):
            print(f"  - {class_name}: {count} images")
        print(f"{'='*50}\n")
