import torch
import platform
import random
import numpy as np
from typing import Optional


class RuntimeConfig:
    """
    A class to configure and manage runtime environment settings.
    
    Automatically detects the operating system and available GPU acceleration,
    then configures PyTorch to use the optimal device.
    """
    
    def __init__(self, seed: int = 42, verbose: bool = True):
        """
        Initialize the RuntimeConfig.
        
        Args:
            seed: Random seed for reproducibility
            verbose: Whether to print detailed device information
        """
        self.seed = seed
        self.verbose = verbose
        self.os_name = platform.system()
        self.device = None
        self.device_name = None
        
    def _set_seed(self) -> None:
        """Set random seeds for reproducibility across all libraries."""
        # Python built-in random
        random.seed(self.seed)
        
        # NumPy
        np.random.seed(self.seed)
        
        # PyTorch
        torch.manual_seed(self.seed)
        
        # CUDA (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set environment variables for additional reproducibility
        import os
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # For DataLoader workers (if num_workers > 0)
        def seed_worker(worker_id):
            worker_seed = self.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        # Store seed_worker function for DataLoader usage
        self.seed_worker = seed_worker
        
        if self.verbose:
            print(f"✓ Random seed set to: {self.seed}")
            print(f"✓ Deterministic mode enabled for reproducibility")
    
    def _detect_device(self) -> torch.device:
        """
        Detect and return the best available device.
        
        Returns:
            torch.device: The selected device (cuda, mps, or cpu)
        """
        device_info = {
            'os': self.os_name,
            'device_type': None,
            'device_name': None,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
        
        # Check for CUDA (Windows/Linux)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_info['device_type'] = 'cuda'
            device_info['device_name'] = torch.cuda.get_device_name(0)
            device_info['cuda_version'] = torch.version.cuda
            device_info['gpu_count'] = torch.cuda.device_count()
        
        # Check for MPS (macOS)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            device_info['device_type'] = 'mps'
            device_info['device_name'] = 'Apple Silicon GPU (Metal Performance Shaders)'
        
        # Fallback to CPU
        else:
            device = torch.device('cpu')
            device_info['device_type'] = 'cpu'
            device_info['device_name'] = 'CPU'
        
        self.device_info = device_info
        return device
    
    def _print_device_info(self) -> None:
        """Print detailed information about the selected device."""
        print(f"\n{'='*60}")
        print("Runtime Configuration")
        print(f"{'='*60}")
        print(f"Operating System: {self.device_info['os']}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Device Type: {self.device_info['device_type'].upper()}")
        print(f"Device Name: {self.device_info['device_name']}")
        
        if self.device_info['device_type'] == 'cuda':
            print(f"CUDA Version: {self.device_info['cuda_version']}")
            print(f"GPU Count: {self.device_info['gpu_count']}")
            
            # Print GPU memory info
            for i in range(self.device_info['gpu_count']):
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({total_memory:.2f} GB)")
        
        elif self.device_info['device_type'] == 'mps':
            print("MPS Backend: Enabled")
            print("Note: Using Apple Silicon GPU acceleration")
        
        else:
            print("Note: No GPU acceleration available. Using CPU.")
            print("      For faster training, consider using a GPU-enabled device.")
        
        print(f"{'='*60}\n")
    
    def setup(self) -> torch.device:
        """
        Setup the runtime environment.
        
        Returns:
            torch.device: The configured device to use for computations
        """
        # Set random seeds
        self._set_seed()
        
        # Detect and configure device
        self.device = self._detect_device()
        self.device_name = self.device_info['device_name']
        
        # Print information if verbose
        if self.verbose:
            self._print_device_info()
        
        return self.device
    
    def get_device(self) -> torch.device:
        """
        Get the configured device.
        
        Returns:
            torch.device: The current device
        """
        if self.device is None:
            raise RuntimeError("Device not configured. Call setup() first.")
        return self.device
    
    def get_device_info(self) -> dict:
        """
        Get detailed device information.
        
        Returns:
            dict: Dictionary containing device information
        """
        if self.device is None:
            raise RuntimeError("Device not configured. Call setup() first.")
        return self.device_info
    
    def to_device(self, tensor_or_model):
        """
        Move a tensor or model to the configured device.
        
        Args:
            tensor_or_model: PyTorch tensor or model to move
            
        Returns:
            The tensor or model on the configured device
        """
        if self.device is None:
            raise RuntimeError("Device not configured. Call setup() first.")
        return tensor_or_model.to(self.device)