#!/usr/bin/env python3
"""
Download Cats vs Dogs dataset.
- Kaggle API: Set KAGGLE_USERNAME, KAGGLE_KEY or place kaggle.json in ~/.kaggle/
- Fallback: Creates sample 224x224 images for pipeline testing (CI/local dev)
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """Create necessary directory structure."""
    dirs = ["data/raw", "data/processed", "models", "logs", "mlruns"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory: {dir_path}")


def download_via_kaggle():
    """Download from Kaggle. Dataset: tongpython/cat-and-dog or similar."""
    try:
        import ssl

        # Fix SSL on corporate networks: KAGGLE_SSL_VERIFY=0 skips cert verification
        if os.environ.get("KAGGLE_SSL_VERIFY", "1") == "0":
            ssl._create_default_https_context = ssl._create_unverified_context

        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "tongpython/cat-and-dog",
            path="data/raw",
            unzip=True,
        )
        logger.info("✓ Downloaded via Kaggle API")
        return True
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        logger.info("Using sample dataset instead.")
        return False


def create_sample_dataset(n_per_class=200):
    """Create sample 224x224 RGB images for pipeline/CI testing."""
    import numpy as np
    from PIL import Image

    raw_dir = Path("data/raw/cats_vs_dogs")
    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            (raw_dir / split / cls).mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    total = n_per_class * 2
    for i in range(total):
        cls = "dogs" if i % 2 == 0 else "cats"
        # Slightly different patterns per class for learnable signal
        base = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        if cls == "cats":
            base[:, :100] = np.clip(base[:, :100] + 30, 0, 255).astype(np.uint8)
        arr = np.clip(base + np.random.randn(224, 224, 3).astype(np.int16) * 5, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        # Assign to split: 80/10/10
        r = i % 100
        split = "train" if r < 80 else ("val" if r < 90 else "test")
        out_path = raw_dir / split / cls / f"{i:05d}.jpg"
        img.save(out_path)

    logger.info(f"✓ Sample dataset: {total} images in data/raw/cats_vs_dogs")
    return True


def main():
    logger.info("=" * 60)
    logger.info("Cats vs Dogs - Data Download")
    logger.info("=" * 60)
    ensure_directories()

    if not download_via_kaggle():
        create_sample_dataset()

    logger.info("Next: python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
