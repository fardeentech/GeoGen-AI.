"""Train a real-image land-use classifier for downstream evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import models
from tqdm import tqdm

from dataset_utils import get_dataloaders, prepare_eurosat
from models import Generator

ROOT = Path(__file__).resolve().parent


class TensorLabelDataset(Dataset):
    """Wrap an image dataset so labels are tensors, matching generated samples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, torch.as_tensor(label, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gan-checkpoint", help="include generated tiles in training; omit for real-only baseline")
    parser.add_argument("--synthetic-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_dir = Path(args.data_dir) if args.data_dir else prepare_eurosat()
    train_loader, val_loader = get_dataloaders(data_dir, args.batch_size, args.workers)
    class_names = train_loader.dataset.dataset.classes
    training_mode = "real_only"
    if args.gan_checkpoint:
        if args.synthetic_per_class < 1:
            parser.error("--synthetic-per-class must be at least 1")
        gan = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
        saved_classes = gan.get("class_names") or class_names
        if saved_classes != class_names:
            parser.error("GAN checkpoint class order does not match the dataset class order")
        generator = Generator(latent_dim=gan["latent_dim"]).to(device)
        generator.load_state_dict(gan["generator"])
        generator.eval()
        synthetic_images, synthetic_labels = [], []
        with torch.no_grad():
            for class_id in range(len(class_names)):
                remaining = args.synthetic_per_class
                while remaining:
                    batch = min(args.batch_size, remaining)
                    labels = torch.full((batch,), class_id, dtype=torch.long, device=device)
                    noise = torch.randn(batch, gan["latent_dim"], device=device)
                    synthetic_images.append(generator(noise, labels).cpu())
                    synthetic_labels.append(labels.cpu())
                    remaining -= batch
        synthetic_dataset = TensorDataset(torch.cat(synthetic_images), torch.cat(synthetic_labels))
        combined_dataset = ConcatDataset([TensorLabelDataset(train_loader.dataset), synthetic_dataset])
        train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=torch.cuda.is_available())
        training_mode = "real_plus_synthetic"
    # Keep model initialization matched for baseline and augmented comparisons.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = count = 0
        for images, labels in tqdm(train_loader, desc=f"Classifier {epoch}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
        model.eval()
        val_correct = val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                val_correct += (logits.argmax(1).cpu() == labels).sum().item()
                val_count += labels.size(0)
        row = {"epoch": epoch, "train_loss": total_loss / count,
               "train_accuracy": correct / count, "validation_accuracy": val_correct / val_count}
        history.append(row)
        print(json.dumps(row))
    out = ROOT / "artifacts" / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "eurosat_classifier_augmented.pt" if training_mode == "real_plus_synthetic" else "eurosat_classifier_real_only.pt"
    torch.save({"model": model.state_dict(), "class_names": class_names,
                "training_mode": training_mode, "seed": args.seed,
                "synthetic_per_class": args.synthetic_per_class if args.gan_checkpoint else 0},
               out / checkpoint_name)
    metrics = ROOT / "artifacts" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / f"classifier_training_{training_mode}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved {training_mode} checkpoint: {out / checkpoint_name}")


if __name__ == "__main__":
    main()

