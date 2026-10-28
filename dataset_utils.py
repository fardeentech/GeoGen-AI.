"""EuroSAT RGB download, extraction, validation and preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
EXPECTED_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_eurosat(force: bool = False) -> Path:
    """Download the authors' RGB archive from its official Zenodo record."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "EuroSAT_RGB.zip"
    if archive.exists() and not force:
        if _md5(archive) != EXPECTED_MD5:
            raise ValueError(f"Checksum mismatch for {archive}; remove it and download again.")
        return archive
    temporary = archive.with_suffix(".zip.part")
    try:
        with urlopen(EUROSAT_URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if _md5(temporary) != EXPECTED_MD5:
            raise ValueError("Downloaded archive checksum did not match Zenodo's published MD5.")
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract only paths that remain under destination."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        zipped.extractall(destination)


def prepare_eurosat(force_download: bool = False) -> Path:
    """Download and extract RGB images; return the class-folder directory."""
    archive = download_eurosat(force=force_download)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _safe_extract(archive, RAW_DIR)
    candidates = [p for p in RAW_DIR.rglob("2750") if p.is_dir()]
    if candidates:
        dataset_dir = candidates[0]
    else:
        candidates = [p for p in RAW_DIR.rglob("EuroSAT_RGB") if p.is_dir()]
        dataset_dir = candidates[0] if candidates else RAW_DIR
    classes = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if len(classes) != 10:
        raise RuntimeError(f"Expected 10 class folders under {dataset_dir}, found {len(classes)}. Inspect the archive layout.")
    return dataset_dir


def get_dataloaders(data_dir: str | Path, batch_size: int = 64, num_workers: int = 2,
                    image_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Create deterministic train/validation splits normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(str(data_dir), transform=transform)
    if len(dataset.classes) != 10:
        raise ValueError(f"Expected 10 classes; found {len(dataset.classes)}: {dataset.classes}")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch_random_split(dataset, [train_size, val_size])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                   pin_memory=torch.cuda.is_available()),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=torch.cuda.is_available()),
    )


def torch_random_split(dataset, lengths):
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.random_split(dataset, lengths, generator=generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download the archive again")
    args = parser.parse_args()
    print(f"Dataset ready at: {prepare_eurosat(force_download=args.force)}")

