"""Fast CPU smoke tests; use tiny temporary images and never download EuroSAT."""
from __future__ import annotations

import argparse
import hashlib
import io
import ast
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
import app
import dataset_utils
import evaluate
import train_classifier
import train_gan
from models import CLASS_NAMES, Discriminator, Generator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_tiny_dataset(root: Path, images_per_class: int = 1) -> Path:
    """Create a minimal ImageFolder-shaped fixture (no network or real data)."""
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_id in range(images_per_class):
            color = (class_id * 20 % 256, image_id * 80 % 256, 100)
            Image.new("RGB", (64, 64), color).save(class_dir / f"sample_{image_id}.png")
    return root


class GeoGenSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_torch_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.previous_torch_threads))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_torch_threads)

    def test_conditional_models_have_expected_shapes_and_range(self):
        generator = Generator(latent_dim=16).eval()
        discriminator = Discriminator().eval()
        noise = torch.randn(2, 16)
        labels = torch.tensor([0, 9], dtype=torch.long)
        with torch.no_grad():
            images = generator(noise, labels)
            scores = discriminator(images, labels)
        self.assertEqual(tuple(images.shape), (2, 3, 64, 64))
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(images.min().item(), -1.0)
        self.assertLessEqual(images.max().item(), 1.0)
        self.assertEqual(tuple(scores.shape), (2,))
        png = Image.open(io.BytesIO(app.image_to_png(((images[0] + 1) / 2).clamp(0, 1))))
        self.assertEqual(png.format, "PNG")
        self.assertEqual(png.size, (64, 64))

    def test_colab_notebook_is_valid_and_saves_app_compatible_checkpoint(self):
        notebook_path = PROJECT_ROOT / "notebooks" / "colab_training.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        python_cells = []
        all_source = []
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            all_source.append(source)
            if cell.get("cell_type") != "code":
                continue
            python_only = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            if python_only.strip():
                ast.parse(python_only)
                python_cells.append(python_only)
        source_text = "\n".join(all_source)
        self.assertIn("train_gan.py", source_text)
        self.assertIn("conditional_dcgan_latest.pt", source_text)
        self.assertIn("Generator(latent_dim=payload['latent_dim'])", source_text)
        self.assertIn("payload['generator']", source_text)
        self.assertIn("files.download(str(CHECKPOINT_PATH))", source_text)
        self.assertGreaterEqual(len(python_cells), 4)

    def test_dataset_loader_reads_tiny_class_folders_and_normalizes(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            dataset_dir = make_tiny_dataset(Path(temporary), images_per_class=2)
            train_loader, validation_loader = dataset_utils.get_dataloaders(
                dataset_dir, batch_size=4, num_workers=0
            )
            images, labels = next(iter(train_loader))
            self.assertEqual(tuple(images.shape[1:]), (3, 64, 64))
            self.assertEqual(len(train_loader.dataset) + len(validation_loader.dataset), 20)
            self.assertTrue(images.min() >= -1 and images.max() <= 1)
            self.assertTrue(set(labels.tolist()).issubset(set(range(10))))

    def test_dataset_download_checksum_and_safe_extraction_without_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for class_name in CLASS_NAMES:
                    archive.writestr(f"2750/{class_name}/", "")
            archive_bytes = buffer.getvalue()
            expected_md5 = hashlib.md5(archive_bytes).hexdigest()

            with patch.object(dataset_utils, "RAW_DIR", root / "raw"), \
                 patch.object(dataset_utils, "EXPECTED_MD5", expected_md5), \
                 patch.object(dataset_utils, "urlopen", return_value=io.BytesIO(archive_bytes)):
                found_dir = dataset_utils.prepare_eurosat()
            self.assertEqual(found_dir.name, "2750")
            self.assertEqual(sorted(path.name for path in found_dir.iterdir()), sorted(CLASS_NAMES))
            self.assertEqual(dataset_utils._md5(root / "raw/EuroSAT_RGB.zip"), expected_md5)

            bad_zip = root / "unsafe.zip"
            with zipfile.ZipFile(bad_zip, "w") as archive:
                archive.writestr("../outside.txt", "must not extract")
            with self.assertRaises(ValueError):
                dataset_utils._safe_extract(bad_zip, root / "safe")
            self.assertFalse((root / "outside.txt").exists())

    def test_one_batch_training_checkpoint_loads_in_app_and_evaluation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            dataset_dir = make_tiny_dataset(root / "tiny_eurosat", images_per_class=1)
            artifacts = root / "artifacts"
            checkpoint_dir = artifacts / "checkpoints"
            checkpoint_dir.mkdir(parents=True)

            # One tiny CPU update proves the trainer writes an app-compatible checkpoint.
            args = argparse.Namespace(
                seed=7, cpu=True, data_dir=str(dataset_dir), batch_size=9,
                workers=0, latent_dim=16, lr=0.0002, epochs=1,
            )
            with patch.object(train_gan, "ROOT", root):
                train_gan.train(args)
            gan_path = checkpoint_dir / "conditional_dcgan_latest.pt"
            self.assertTrue(gan_path.exists())
            checkpoint = torch.load(gan_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["class_names"], CLASS_NAMES)

            # Exercise the same checkpoint-loading function used by Streamlit.
            generator, loaded, device = app.load_generator(str(gan_path), gan_path.stat().st_mtime)
            with torch.no_grad():
                result = generator(torch.randn(1, loaded["latent_dim"], device=device), torch.tensor([0], device=device))
            self.assertEqual(tuple(result.shape), (1, 3, 64, 64))

            # Train tiny real-only and augmented classifiers, then run the comparison path.
            with patch.object(train_classifier, "ROOT", root), patch("sys.argv", [
                "train_classifier.py", "--data-dir", str(dataset_dir), "--epochs", "1",
                "--batch-size", "32", "--workers", "0", "--cpu", "--seed", "7",
            ]):
                train_classifier.main()
            with patch.object(train_classifier, "ROOT", root), patch("sys.argv", [
                "train_classifier.py", "--data-dir", str(dataset_dir), "--epochs", "1",
                "--batch-size", "32", "--workers", "0", "--cpu", "--seed", "7",
                "--gan-checkpoint", str(gan_path), "--synthetic-per-class", "1",
            ]):
                train_classifier.main()

            with patch.object(evaluate, "ROOT", root), patch("sys.argv", [
                "evaluate.py", "--data-dir", str(dataset_dir), "--gan-checkpoint", str(gan_path),
                "--baseline-checkpoint", str(checkpoint_dir / "eurosat_classifier_real_only.pt"),
                "--augmented-checkpoint", str(checkpoint_dir / "eurosat_classifier_augmented.pt"),
                "--samples-per-class", "1",
            ]):
                evaluate.main()
            self.assertTrue((root / "artifacts/metrics/evaluation_comparison.json").exists())
            self.assertTrue((root / "artifacts/metrics/accuracy_comparison.png").exists())
            self.assertEqual(len(list((root / "artifacts/generated").glob("*.png"))), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
