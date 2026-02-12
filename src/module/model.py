import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class PrototypicalNetwork(nn.Module):
    """
    Cosine-based Prototypical Network with temperature scaling.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_dims: tuple = (256,),
        pretrained: bool = True,
        cache_dir: str = "../backbones",
        initial_temperature: float = 10.0,
    ):
        super(PrototypicalNetwork, self).__init__()

        self.backbone_name = backbone.lower()
        self.embedding_dims = embedding_dims
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        torch.hub.set_dir(str(self.cache_dir))

        # Load backbone
        self.backbone, self.feature_dim = self._load_backbone(pretrained)

        # Create embedding head
        self.embedding = self._create_embedding_layers()

        # Final embedding dimension
        self.final_embedding_dim = (
            embedding_dims[-1] if embedding_dims else self.feature_dim
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    # --------------------------------------------------
    # Backbone Loading
    # --------------------------------------------------
    def _load_backbone(self, pretrained: bool):

        if self.backbone_name == "resnet50":
            model = torch.hub.load(
                "pytorch/vision:v0.10.0",
                "resnet50",
                pretrained=pretrained,
                verbose=False,
            )
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim

        elif self.backbone_name in ["vit", "vit_b_16"]:
            from torchvision.models import vit_b_16, ViT_B_16_Weights

            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = vit_b_16(weights=weights)

            feature_dim = model.heads.head.in_features
            model.heads.head = nn.Identity()

            return model, feature_dim

        elif self.backbone_name in ["dinov2", "dinov2_vits14"]:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=pretrained,
                verbose=False,
            )
            feature_dim = 384
            return model, feature_dim

        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                f"Supported: resnet50, vit, dinov2"
            )

    # --------------------------------------------------
    # Embedding Head
    # --------------------------------------------------
    def _create_embedding_layers(self):

        if not self.embedding_dims:
            return nn.Identity()

        layers = []
        input_dim = self.feature_dim

        for i, output_dim in enumerate(self.embedding_dims):

            layers.append(nn.Linear(input_dim, output_dim))

            if i < len(self.embedding_dims) - 1:
                layers.append(nn.BatchNorm1d(output_dim))
                layers.append(nn.ReLU(inplace=True))

            input_dim = output_dim

        return nn.Sequential(*layers)

    # --------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:

        features = self.backbone(x)
        embeddings = self.embedding(features)

        # Normalize to unit hypersphere
        embeddings = F.normalize(embeddings, dim=1)

        return embeddings

    # --------------------------------------------------
    # Prototype Computation
    # --------------------------------------------------
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:

        support_embeddings = support_embeddings.view(n_way, k_shot, -1)
        prototypes = support_embeddings.mean(dim=1)

        # Normalize prototypes for cosine geometry
        prototypes = F.normalize(prototypes, dim=1)

        return prototypes

    # --------------------------------------------------
    # Cosine Similarity
    # --------------------------------------------------
    def compute_similarities(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:

        # Dot product = cosine similarity (since normalized)
        similarities = torch.matmul(query_embeddings, prototypes.t())

        return similarities

    # --------------------------------------------------
    # Forward Episode
    # --------------------------------------------------
    def forward(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int,
        k_shot: int,
    ):

        support_embeddings = self.extract_features(support_images)
        query_embeddings = self.extract_features(query_images)

        prototypes = self.compute_prototypes(
            support_embeddings,
            n_way,
            k_shot,
        )

        similarities = self.compute_similarities(
            query_embeddings,
            prototypes,
        )

        # Clamp temperature to prevent instability
        temperature = torch.clamp(self.temperature, 1.0, 50.0)

        logits = similarities * temperature

        return logits, prototypes

    # --------------------------------------------------
    # Backbone Control
    # --------------------------------------------------
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    # --------------------------------------------------
    # Info
    # --------------------------------------------------
    def get_model_info(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "backbone": self.backbone_name,
            "feature_dim": self.feature_dim,
            "embedding_dims": self.embedding_dims,
            "final_embedding_dim": self.final_embedding_dim,
            "temperature": self.temperature.item(),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_frozen": not next(
                self.backbone.parameters()
            ).requires_grad,
        }

    def print_model_info(self):

        info = self.get_model_info()

        print(f"\n{'='*70}")
        print("Cosine Prototypical Network")
        print(f"{'='*70}")
        print(f"Backbone: {info['backbone']}")
        print(f"Feature Dim: {info['feature_dim']}")
        print(f"Embedding Dims: {info['embedding_dims']}")
        print(f"Final Embedding Dim: {info['final_embedding_dim']}")
        print(f"Temperature: {info['temperature']:.4f}")
        print(f"Total Params: {info['total_parameters']:,}")
        print(f"Trainable Params: {info['trainable_parameters']:,}")
        print(f"Backbone Frozen: {info['backbone_frozen']}")
        print(f"{'='*70}\n")
