"""Train a class-conditional DCGAN on EuroSAT RGB tiles."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from dataset_utils import get_dataloaders, prepare_eurosat
from models import Discriminator, Generator, initialize_weights

ROOT = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_dir = Path(args.data_dir) if args.data_dir else prepare_eurosat()
    train_loader, _ = get_dataloaders(data_dir, args.batch_size, args.workers)
    generator = Generator(latent_dim=args.latent_dim).to(device).apply(initialize_weights)
    discriminator = Discriminator().to(device).apply(initialize_weights)
    optimizer_g = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_dir = ROOT / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        totals = {"g": 0.0, "d": 0.0}
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for real, labels in progress:
            real, labels = real.to(device), labels.to(device)
            batch_size = real.size(0)
            real_targets = torch.full((batch_size,), 0.9, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real, labels), real_targets)
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake = generator(noise, labels)
            fake_loss = criterion(discriminator(fake.detach(), labels), fake_targets)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake, labels), torch.ones(batch_size, device=device))
            g_loss.backward()
            optimizer_g.step()
            totals["g"] += g_loss.item()
            totals["d"] += d_loss.item()
            progress.set_postfix(g=f"{g_loss.item():.3f}", d=f"{d_loss.item():.3f}")

        row = {"epoch": epoch, "generator_loss": totals["g"] / len(train_loader),
               "discriminator_loss": totals["d"] / len(train_loader)}
        history.append(row)
        checkpoint = {"epoch": epoch, "generator": generator.state_dict(),
                      "discriminator": discriminator.state_dict(), "latent_dim": args.latent_dim,
                      "class_names": train_loader.dataset.dataset.classes if hasattr(train_loader.dataset, "dataset") else [],
                      "seed": args.seed}
        torch.save(checkpoint, checkpoint_dir / "conditional_dcgan_latest.pt")
        (ROOT / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
        (ROOT / "artifacts" / "metrics" / "gan_training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}: G={row['generator_loss']:.4f}, D={row['discriminator_loss']:.4f}")
    print(f"Training complete. Checkpoint: {checkpoint_dir / 'conditional_dcgan_latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="folder with one subfolder per class; downloads EuroSAT when omitted")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU even when CUDA exists")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

