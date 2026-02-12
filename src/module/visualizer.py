from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
import numpy as np
import torch
from pathlib import Path


class EmbeddingVisualizer:
    """
    Visualizer for analyzing model predictions through embedding space visualization.

    Creates 2D scatter plots showing support sets, prototypes, and query predictions
    for each episode to understand model behavior and identify errors.
    """

    def __init__(self, model, test_dataset,
                 device, n_way: int = 2, k_shot: int = 5,
                 query_per_class: int = 5, max_episode: int = 10,
                 plots_dir: str = '../plots', reduction_method: str = 'tsne',
                 jitter_strength: float = 0.02, random_state: int = 42):
        """
        Initialize the embedding visualizer.

        Args:
            model: Trained PrototypicalNetwork instance
            test_dataset: Test dataset
            device: Device to run on
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            query_per_class: Number of query examples per class
            max_episode: Number of episodes to visualize
            plots_dir: Directory to save plots
            reduction_method: Dimensionality reduction method ('tsne' or 'umap')
            jitter_strength: Amount of random jitter to apply (0.0 to 0.1)
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.test_dataset = test_dataset
        self.device = device

        self.n_way = n_way
        self.k_shot = k_shot
        self.query_per_class = query_per_class
        self.max_episode = max_episode

        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.reduction_method = reduction_method.lower()
        self.jitter_strength = jitter_strength
        self.random_state = random_state

        # Color scheme
        self.class_colors = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue for BK, Orange for FK
        self.class_names = {0: 'BK', 1: 'FK'}

    def _extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract final embedding vectors from the model.

        Args:
            images: Input images tensor (batch_size, 3, H, W)

        Returns:
            Embedding vectors (batch_size, embedding_dim)
        """
        self.model.eval()
        with torch.no_grad():
            # Pass through backbone
            features = self.model.backbone(images)

            # Pass through embedding layers
            embeddings = self.model.embedding(features)

        return embeddings

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embeddings to 2D using t-SNE or UMAP.

        Args:
            embeddings: High-dimensional embeddings (n_samples, embedding_dim)

        Returns:
            2D coordinates (n_samples, 2)
        """
        if self.reduction_method == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=self.random_state,
                perplexity=min(30, embeddings.shape[0] - 1),
                n_iter=1000
            )
        elif self.reduction_method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(
                    n_components=2,
                    random_state=self.random_state,
                    n_neighbors=min(15, embeddings.shape[0] - 1)
                )
            except ImportError:
                print("⚠ UMAP not installed. Falling back to t-SNE.")
                print("  Install with: pip install umap-learn")
                reducer = TSNE(
                    n_components=2,
                    random_state=self.random_state,
                    perplexity=min(30, embeddings.shape[0] - 1)
                )
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")

        coords_2d = reducer.fit_transform(embeddings)
        return coords_2d

    def _apply_jitter(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply random jitter to coordinates to prevent overlapping markers.

        Args:
            coords: 2D coordinates (n_samples, 2)

        Returns:
            Jittered coordinates
        """
        np.random.seed(self.random_state)
        jitter = np.random.randn(*coords.shape) * self.jitter_strength
        return coords + jitter

    def _plot_episode(self, episode_id: int, support_coords: np.ndarray,
                     query_coords: np.ndarray, prototype_coords: np.ndarray,
                     support_labels: np.ndarray, query_labels: np.ndarray,
                     query_predictions: np.ndarray, support_indices: list,
                     query_indices: list) -> None:
        """
        Create and save a scatter plot for one episode.

        Args:
            episode_id: Episode number
            support_coords: 2D coords for support samples (n_support, 2)
            query_coords: 2D coords for query samples (n_query, 2)
            prototype_coords: 2D coords for prototypes (n_way, 2)
            support_labels: True labels for support (n_support,)
            query_labels: True labels for query (n_query,)
            query_predictions: Predicted labels for query (n_query,)
            support_indices: Dataset indices for support samples
            query_indices: Dataset indices for query samples
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Apply jitter to all coordinates
        support_coords = self._apply_jitter(support_coords)
        query_coords = self._apply_jitter(query_coords)

        # Plot support set (squares)
        for class_idx in range(self.n_way):
            mask = support_labels == class_idx
            if np.any(mask):
                ax.scatter(
                    support_coords[mask, 0],
                    support_coords[mask, 1],
                    c=self.class_colors[class_idx],
                    marker='s',
                    s=200,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=2,
                    label=f'{self.class_names[class_idx]} Support'
                )

        # Plot prototypes (plus signs) - use smaller markers for legend
        for class_idx in range(self.n_way):
            ax.scatter(
                prototype_coords[class_idx, 0],
                prototype_coords[class_idx, 1],
                c=self.class_colors[class_idx],
                marker='P',
                s=300,  # Reduced from 500 for better legend spacing
                edgecolors='black',
                linewidths=2,  # Reduced from 3
                label=f'{self.class_names[class_idx]} Centroid',
                zorder=10
            )

        # Plot query set (circles, colored by prediction)
        for class_idx in range(self.n_way):
            mask = query_predictions == class_idx
            if np.any(mask):
                ax.scatter(
                    query_coords[mask, 0],
                    query_coords[mask, 1],
                    c=self.class_colors[class_idx],
                    marker='o',
                    s=150,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=1.5,
                    label=f'Predicted as {self.class_names[class_idx]}'
                )

        # Add sample ID labels for support set (above the markers with offset)
        texts = []
        y_offset = 0.02 * (np.max(support_coords[:, 1]) - np.min(support_coords[:, 1]))  # 2% of y-range
        for i, (x, y) in enumerate(support_coords):
            sample_id = support_indices[i] + 1  # Row number starts from 1 (row 0 is header)
            # Position label above the marker with vertical offset
            texts.append(ax.text(x, y + y_offset, f'S{sample_id}', fontsize=9, ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='gray', alpha=0.8, linewidth=0.5),
                               fontweight='bold'))

        # Add sample ID labels for query set (above the markers with offset)
        for i, (x, y) in enumerate(query_coords):
            sample_id = query_indices[i] + 1  # Row number starts from 1 (row 0 is header)
            # Mark misclassifications with asterisk
            true_label = query_labels[i]
            pred_label = query_predictions[i]
            marker = '*' if true_label != pred_label else ''
            # Get true class name
            true_class_name = self.class_names[true_label]
            # Position label above the marker with vertical offset
            label_text = f'Q{sample_id} ({true_class_name}){marker}'
            texts.append(ax.text(x, y + y_offset, label_text, fontsize=9,
                               ha='center', va='bottom',
                               color='red' if marker else 'black',
                               bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white',
                                       edgecolor='red' if marker else 'gray',
                                       alpha=0.9 if marker else 0.8,
                                       linewidth=1.0 if marker else 0.5),
                               fontweight='bold'))

        # Adjust text positions to avoid overlap
        try:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except:
            # If adjust_text fails, continue without adjustment
            pass

        # Formatting
        ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Episode {episode_id}: Embedding Space Visualization '
            f'({self.reduction_method.upper()})\n'
            f'{self.n_way}-way {self.k_shot}-shot | '
            f'Backbone: {self.model.backbone_name}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Create a better legend for manuscript with proper spacing
        ax.legend(
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            fontsize=10,
            frameon=True,
            framealpha=0.95,
            edgecolor='black',
            fancybox=True,
            shadow=True,
            ncol=1,
            columnspacing=2.0,
            labelspacing=1.5,        # Even more spacing between entries
            handletextpad=1.2,       # More space between marker and text
            handlelength=3.0,        # Longer marker line
            borderpad=1.5,           # More padding inside legend box
            markerscale=0.8          # Slightly smaller markers in legend
        )

        ax.grid(True, alpha=0.3, linestyle='--')

        # Add note about labels and misclassifications
        error_count = np.sum(query_labels != query_predictions)
        note = (f'Labels: S=Support, Q=Query (sample ID from test.csv)\n'
                f'* indicates misclassification ({error_count}/{len(query_labels)} errors)')
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black', linewidth=1))

        plt.tight_layout()

        # Save plot
        filename = f"episode_{episode_id}.svg"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved episode {episode_id} plot: {filepath}")

    def visualize(self, verbose: bool = True) -> None:
        """
        Generate embedding visualizations for multiple test episodes.

        Args:
            verbose: Whether to print progress information
        """
        if verbose:
            print("=" * 70)
            print("Starting Embedding Space Visualization")
            print("=" * 70)
            print(f"Configuration:")
            print(f"  Episodes to visualize: {self.max_episode}")
            print(f"  N-way: {self.n_way}, K-shot: {self.k_shot}, Query per class: {self.query_per_class}")
            print(f"  Reduction method: {self.reduction_method.upper()}")
            print(f"  Device: {self.device}")
            print(f"  Output directory: {self.plots_dir}")
            print("=" * 70)

        self.model.eval()

        # Build class-to-indices mapping
        class_to_indices = {}
        for idx in range(len(self.test_dataset)):
            _, label, _, _ = self.test_dataset[idx]
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        classes = list(class_to_indices.keys())

        # Validate
        if len(classes) < self.n_way:
            raise ValueError(f"Test dataset has {len(classes)} classes but n_way={self.n_way}")

        for episode_id in range(1, self.max_episode + 1):
            if verbose:
                print(f"\nProcessing Episode {episode_id}/{self.max_episode}...")

            # Manually sample episode
            # Randomly select N classes for this episode
            selected_classes = np.random.choice(classes, self.n_way, replace=False)

            support_indices = []
            query_indices = []

            for cls in selected_classes:
                # Get all indices for this class
                cls_indices = class_to_indices[cls].copy()

                # Randomly sample K+Q indices from this class
                sampled_indices = np.random.choice(
                    cls_indices,
                    self.k_shot + self.query_per_class,
                    replace=False
                )

                # First K samples are support, rest are query
                support_indices.extend(sampled_indices[:self.k_shot])
                query_indices.extend(sampled_indices[self.k_shot:])

            # Get data
            support_data = [self.test_dataset[i] for i in support_indices]
            query_data = [self.test_dataset[i] for i in query_indices]

            support_images = torch.stack([x[0] for x in support_data]).to(self.device)
            support_labels = torch.tensor([x[1] for x in support_data])
            support_filenames = [x[2] for x in support_data]

            query_images = torch.stack([x[0] for x in query_data]).to(self.device)
            query_labels = torch.tensor([x[1] for x in query_data])
            query_filenames = [x[2] for x in query_data]

            # Extract embeddings
            support_embeddings = self._extract_embeddings(support_images)
            query_embeddings = self._extract_embeddings(query_images)

            # Compute prototypes in high-dimensional space (for predictions)
            support_embeddings_reshaped = support_embeddings.view(
                self.n_way, self.k_shot, -1
            )
            prototypes = support_embeddings_reshaped.mean(dim=1)  # (n_way, embedding_dim)

            # Get predictions
            with torch.no_grad():
                dists = torch.cdist(query_embeddings, prototypes)  # (n_query, n_way)
                query_predictions = torch.argmin(dists, dim=1)

            # Combine support and query embeddings for dimensionality reduction
            # NOTE: We reduce only support and query, then compute centroids in 2D space
            all_embeddings = torch.cat([
                support_embeddings,
                query_embeddings
            ], dim=0).cpu().numpy()

            # Reduce to 2D
            coords_2d = self._reduce_dimensions(all_embeddings)

            # Split back into components
            n_support = len(support_embeddings)
            n_query = len(query_embeddings)

            support_coords = coords_2d[:n_support]
            query_coords = coords_2d[n_support:n_support + n_query]

            # Compute prototype coordinates in 2D space from reduced support coordinates
            support_coords_reshaped = support_coords.reshape(self.n_way, self.k_shot, 2)
            prototype_coords = support_coords_reshaped.mean(axis=1)  # (n_way, 2)

            # Create plot
            self._plot_episode(
                episode_id=episode_id,
                support_coords=support_coords,
                query_coords=query_coords,
                prototype_coords=prototype_coords,
                support_labels=support_labels.numpy(),
                query_labels=query_labels.numpy(),
                query_predictions=query_predictions.cpu().numpy(),
                support_indices=support_indices,
                query_indices=query_indices
            )

        if verbose:
            print("\n" + "=" * 70)
            print("Visualization Complete!")
            print(f"Generated {self.max_episode} episode plots in {self.plots_dir}")
            print("=" * 70)
