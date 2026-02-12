from torchvision import transforms
import torch

class ImageTransforms:
    """
    A class to create image transformation pipelines for training and testing.
    """
    
    def __init__(self, image_size: int = 224, augment_training: bool = True):
        """
        Initialize the ImageTransforms.
        
        Args:
            image_size: Target size for images (default: 224 for most CNNs)
            augment_training: Whether to apply augmentation to training transforms
        """
        self.image_size = image_size
        self.augment_training = augment_training
        
        # ImageNet normalization values (standard for transfer learning)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.train_transform = self._create_train_transform()
        self.test_transform = self._create_test_transform()
    
    def _create_train_transform(self):
        """Create transformation pipeline for training data."""
        if self.augment_training:
            transform = transforms.Compose([
                # Resize slightly larger before cropping
                transforms.Resize(int(self.image_size * 1.15)),
                
                # Random crop
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),

                # 🔥 Elastic Deformation (NEW)
                transforms.ElasticTransform(
                    alpha=40.0,     # deformation intensity
                    sigma=5.0,      # smoothness of displacement
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0
                ),

                # Standard augmentations
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                
                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        
        return transform
    
    def _create_test_transform(self):
        """Create transformation pipeline for test/validation data."""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return transform
    
    def get_train_transform(self):
        return self.train_transform
    
    def get_test_transform(self):
        return self.test_transform
    
    def denormalize(self, tensor):
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return tensor * std + mean
    
    def print_info(self):
        print(f"\n{'='*60}")
        print("Image Transforms Information")
        print(f"{'='*60}")
        print(f"Image Size: {self.image_size}x{self.image_size}")
        print(f"Training Augmentation: {'Enabled' if self.augment_training else 'Disabled'}")
        print(f"Normalization Mean: {self.mean}")
        print(f"Normalization Std: {self.std}")
        print(f"\nTraining Transforms:")
        print(self.train_transform)
        print(f"\nTest/Val Transforms:")
        print(self.test_transform)
        print(f"{'='*60}\n")
