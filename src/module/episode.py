from torch.utils.data import Sampler
import numpy as np
from typing import Tuple, Dict
from module.dataset import ImageDataset

class EpisodeSampler(Sampler):
    """
    Sampler for few-shot learning episodes (N-way K-shot).
    
    Generates episodes where each episode contains:
    - Support set: K examples per N classes
    - Query set: Q examples per N classes
    """
    
    def __init__(self, dataset: ImageDataset, n_way: int, k_shot: int, 
                 query_per_class: int, n_episodes: int):
        """
        Initialize the EpisodeSampler.
        
        Args:
            dataset: ImageDataset instance
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            query_per_class: Number of query examples per class
            n_episodes: Total number of episodes to generate
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.n_episodes = n_episodes
        
        # Get indices for each class
        self.class_to_indices = self._create_class_index_map()
        self.classes = list(self.class_to_indices.keys())
        
        # Validate that we have enough classes
        if len(self.classes) < n_way:
            raise ValueError(f"Dataset has {len(self.classes)} classes but n_way={n_way}")
        
        # Validate that each class has enough samples
        min_samples_needed = k_shot + query_per_class
        for cls, indices in self.class_to_indices.items():
            if len(indices) < min_samples_needed:
                raise ValueError(
                    f"Class {cls} has {len(indices)} samples but needs at least "
                    f"{min_samples_needed} ({k_shot} support + {query_per_class} query)"
                )
    
    def _create_class_index_map(self) -> Dict[int, list]:
        """Create a mapping from class labels to dataset indices."""
        class_to_indices = {}
        for idx in range(len(self.dataset)):
            _, label, _, _ = self.dataset[idx]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def __len__(self) -> int:
        """Return the number of episodes."""
        return self.n_episodes
    
    def __iter__(self):
        """Generate episodes."""
        for _ in range(self.n_episodes):
            # Randomly select N classes for this episode
            selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
            
            support_indices = []
            query_indices = []
            
            for cls in selected_classes:
                # Get all indices for this class
                cls_indices = self.class_to_indices[cls].copy()
                
                # Randomly sample K+Q indices from this class
                sampled_indices = np.random.choice(
                    cls_indices, 
                    self.k_shot + self.query_per_class, 
                    replace=False
                )
                
                # Split into support and query
                support_indices.extend(sampled_indices[:self.k_shot])
                query_indices.extend(sampled_indices[self.k_shot:])
            
            # Yield support indices followed by query indices
            yield support_indices + query_indices


class FewShotBatchSampler:
    """
    Batch sampler that creates batches of few-shot learning episodes.
    """
    
    def __init__(self, dataset: ImageDataset, n_way: int, k_shot: int,
                 query_per_class: int, n_episodes: int, episodes_per_batch: int = 1):
        """
        Initialize the FewShotBatchSampler.
        
        Args:
            dataset: ImageDataset instance
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            query_per_class: Number of query examples per class
            n_episodes: Total number of episodes
            episodes_per_batch: Number of episodes per batch
        """
        self.episode_sampler = EpisodeSampler(
            dataset, n_way, k_shot, query_per_class, n_episodes
        )
        self.episodes_per_batch = episodes_per_batch
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.episode_sampler) // self.episodes_per_batch
    
    def __iter__(self):
        """Generate batches of episodes."""
        batch = []
        for episode in self.episode_sampler:
            batch.append(episode)
            if len(batch) == self.episodes_per_batch:
                # For single episode per batch, yield the episode directly (not wrapped in a list)
                if self.episodes_per_batch == 1:
                    yield batch[0]
                else:
                    yield batch
                batch = []
        
        # Yield remaining episodes if any
        if batch:
            if self.episodes_per_batch == 1:
                yield batch[0]
            else:
                yield batch
    
    def get_episode_info(self) -> dict:
        """Get information about episode configuration."""
        return {
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'query_per_class': self.query_per_class,
            'support_size': self.n_way * self.k_shot,
            'query_size': self.n_way * self.query_per_class,
            'total_samples_per_episode': self.n_way * (self.k_shot + self.query_per_class)
        }