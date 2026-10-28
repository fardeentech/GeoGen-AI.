"""Compare real-only and synthetic-augmented classifiers on the same held-out data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torchvision import models
from torchvision.utils import save_image

from dataset_utils import get_dataloaders, prepare_eurosat
from models import Generator

ROOT = Path(__file__).resolve().parent


def load_classifier(checkpoint_path: str | Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = payload["class_names"]
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, len(class_names))
    classifier.load_state_dict(payload["model"])
    return classifier.to(device).eval(), payload


def measure_classifier(classifier, test_loader, class_names, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = classifier(images.to(device)).argmax(1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "classification_report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--gan-checkpoint", default=str(ROOT / "artifacts/checkpoints/conditional_dcgan_latest.pt"))
    parser.add_argument("--baseline-checkpoint", default=str(ROOT / "artifacts/checkpoints/eurosat_classifier_real_only.pt"))
    parser.add_argument("--augmented-checkpoint", default=str(ROOT / "artifacts/checkpoints/eurosat_classifier_augmented.pt"))
    parser.add_argument("--samples-per-class", type=int, default=16,
                        help="number of generated tiles to export for inspection")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir) if args.data_dir else prepare_eurosat()
    _, test_loader = get_dataloaders(data_dir, batch_size=128, num_workers=2)

    baseline, baseline_payload = load_classifier(args.baseline_checkpoint, device)
    augmented, augmented_payload = load_classifier(args.augmented_checkpoint, device)
    class_names = baseline_payload["class_names"]
    if augmented_payload["class_names"] != class_names:
        parser.error("Baseline and augmented checkpoints use different class orders")
    gan = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
    if gan.get("class_names") and gan["class_names"] != class_names:
        parser.error("GAN checkpoint class order does not match the classifier checkpoints")
    if baseline_payload.get("training_mode") not in (None, "real_only"):
        parser.error("Baseline checkpoint was not trained in real-only mode")
    if augmented_payload.get("training_mode") not in (None, "real_plus_synthetic"):
        parser.error("Augmented checkpoint was not trained with synthetic tiles")
    synthetic_training_count = augmented_payload.get("synthetic_per_class")
    if synthetic_training_count is None:
        parser.error("Augmented classifier checkpoint does not record how many generated images were used")

    baseline_result = measure_classifier(baseline, test_loader, class_names, device)
    augmented_result = measure_classifier(augmented, test_loader, class_names, device)
    comparison = {
        "test_split": "deterministic 10% held-out split from dataset_utils.get_dataloaders",
        "synthetic_per_class_used_for_augmented_training": synthetic_training_count,
        "real_only": baseline_result,
        "real_plus_synthetic": augmented_result,
        "accuracy_change_augmented_minus_baseline": augmented_result["accuracy"] - baseline_result["accuracy"],
    }

    metrics_dir = ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    generated_dir = ROOT / "artifacts" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generator = Generator(latent_dim=gan["latent_dim"]).to(device)
    generator.load_state_dict(gan["generator"])
    generator.eval()
    with torch.no_grad():
        for class_id, class_name in enumerate(class_names):
            count = args.samples_per_class
            for start in range(0, count, 64):
                batch = min(64, count - start)
                labels = torch.full((batch,), class_id, dtype=torch.long, device=device)
                noise = torch.randn(batch, gan["latent_dim"], device=device)
                images = ((generator(noise, labels).cpu() + 1) / 2).clamp(0, 1)
                for offset, image in enumerate(images):
                    save_image(image, generated_dir / f"{class_id}_{class_name}_{start + offset:04d}.png")

    # Two charts compare the actual held-out metrics from the two trained checkpoints.
    accuracies = [baseline_result["accuracy"], augmented_result["accuracy"]]
    plt.figure(figsize=(6, 4))
    plt.bar(["Real only", "Real + synthetic"], accuracies, color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1)
    plt.ylabel("Held-out accuracy")
    plt.title("Classifier accuracy: real-only vs augmented training")
    plt.tight_layout()
    plt.savefig(metrics_dir / "accuracy_comparison.png", dpi=160)
    x = list(range(len(class_names)))
    width = 0.38
    baseline_f1 = [baseline_result["classification_report"][name]["f1-score"] for name in class_names]
    augmented_f1 = [augmented_result["classification_report"][name]["f1-score"] for name in class_names]
    plt.figure(figsize=(11, 5))
    plt.bar([i - width / 2 for i in x], baseline_f1, width, label="Real only")
    plt.bar([i + width / 2 for i in x], augmented_f1, width, label="Real + synthetic")
    plt.xticks(x, class_names, rotation=40, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("F1 score")
    plt.title("Per-class F1 on the same held-out split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_dir / "f1_comparison_by_class.png", dpi=160)
    print(f"Real-only accuracy: {baseline_result['accuracy']:.4f}")
    print(f"Real + synthetic accuracy: {augmented_result['accuracy']:.4f}")
    print(f"Saved comparison metrics to {metrics_dir} and generated samples to {generated_dir}")


if __name__ == "__main__":
    main()

