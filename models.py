"""Conditional DCGAN models for 64x64 EuroSAT RGB tiles."""
from __future__ import annotations

import torch
from torch import nn


NUM_CLASSES = 10
IMAGE_CHANNELS = 3
IMAGE_SIZE = 64
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


class Generator(nn.Module):
    """Maps noise and a class label to a normalized RGB satellite image."""

    def __init__(self, latent_dim: int = 128, num_classes: int = NUM_CLASSES,
                 image_channels: int = IMAGE_CHANNELS, embedding_dim: int = 50):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.project = nn.Linear(latent_dim + embedding_dim, 512 * 4 * 4)
        self.network = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat((noise, self.label_embedding(labels)), dim=1)
        return self.network(self.project(x).view(-1, 512, 4, 4))


class Discriminator(nn.Module):
    """Scores image realism while conditioning on the requested class."""

    def __init__(self, num_classes: int = NUM_CLASSES, image_channels: int = IMAGE_CHANNELS):
        super().__init__()
        self.image_features = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_embedding = nn.Embedding(num_classes, 512)
        self.classifier = nn.Linear(512 * 4 * 4, 1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = self.image_features(images).flatten(1)
        # Projection discriminator: image score plus image/class compatibility.
        class_vector = self.label_embedding(labels)
        pooled = features.view(-1, 512, 4, 4).sum(dim=(2, 3))
        return self.classifier(features).squeeze(1) + (pooled * class_vector).sum(dim=1) / 4


def initialize_weights(module: nn.Module) -> None:
    """DCGAN-style weight initialization."""
    if isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        weight = getattr(module, "weight", None)
        if weight is not None:
            nn.init.normal_(weight, 0.0, 0.02)
        bias = getattr(module, "bias", None)
        if bias is not None:
            nn.init.zeros_(bias)

