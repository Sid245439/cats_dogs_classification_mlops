"""
Preprocessing for Cats vs Dogs images.
- Load images, resize to 224x224 RGB
- Split: train 80% / val 10% / test 10%
- Data augmentation for training
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
CLASSES = ["cat", "dog"]
CLASS_TO_IDX = {"cat": 0, "dog": 1}


def load_image(path: str, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load single image, resize to size, return RGB numpy array [H, W, 3]."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_dataset(
    data_dir: str = "data/raw/cats_vs_dogs",
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load images from train/val/test folders. If not present, create from single folder.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Support structure: data_dir/train/cats/, data_dir/train/dogs/, etc.
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    def _collect_from_folder(folder: Path) -> Tuple[List[np.ndarray], List[int]]:
        X, y = [], []
        if not folder.exists():
            return X, y
        for cls_dir in folder.iterdir():
            if not cls_dir.is_dir():
                continue
            label = 1 if "dog" in cls_dir.name.lower() else 0
            for fp in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png")):
                try:
                    arr = load_image(str(fp))
                    X.append(arr)
                    y.append(label)
                except Exception as e:
                    logger.warning(f"Skip {fp}: {e}")
        return X, y

    X_train, y_train = _collect_from_folder(train_dir)
    X_val, y_val = _collect_from_folder(val_dir)
    X_test, y_test = _collect_from_folder(test_dir)

    if len(X_train) == 0:
        raise ValueError("No images found. Run scripts/download_data.py first.")

    X_train = np.stack(X_train)
    y_train = np.array(y_train, dtype=np.int64)

    if len(X_val) > 0:
        X_val = np.stack(X_val)
        y_val = np.array(y_val, dtype=np.int64)
    else:
        n = len(X_train)
        idx = np.random.RandomState(seed).permutation(n)
        n_val = int(n * splits[1])
        n_test = int(n * splits[2])
        X_val = X_train[idx[n - n_val - n_test : n - n_test]]
        y_val = y_train[idx[n - n_val - n_test : n - n_test]]
        X_test = X_train[idx[n - n_test :]]
        y_test = y_train[idx[n - n_test :]]
        X_train = X_train[idx[: n - n_val - n_test]]
        y_train = y_train[idx[: n - n_val - n_test]]
        return X_train, y_train, X_val, y_val, X_test, y_test

    if len(X_test) > 0:
        X_test = np.stack(X_test)
        y_test = np.array(y_test, dtype=np.int64)
    else:
        n_val = len(X_val)
        n_test = max(1, n_val // 2)
        idx = np.random.RandomState(seed).permutation(n_val)
        X_test = X_val[idx[:n_test]]
        y_test = y_val[idx[:n_test]]
        X_val = X_val[idx[n_test:]]
        y_val = y_val[idx[n_test:]]

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def augment_image(img: np.ndarray) -> np.ndarray:
    """Simple augmentation: horizontal flip, brightness/contrast jitter."""
    img = img.copy()
    if np.random.random() > 0.5:
        img = np.fliplr(img).copy()
    if np.random.random() > 0.5:
        factor = 0.9 + np.random.random() * 0.2
        img = np.clip(img * factor, 0, 1).astype(np.float32)
    return img


def preprocess_for_inference(img_input, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Preprocess image for inference.
    img_input: file path (str), bytes, or numpy array [H,W,3]
    Returns: (1, H, W, 3) float32 in [0,1]
    """
    if isinstance(img_input, (str, Path)):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, bytes):
        from io import BytesIO
        img = Image.open(BytesIO(img_input)).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 2:
            img_input = np.stack([img_input] * 3, axis=-1)
        if img_input.ndim == 3 and img_input.shape[-1] == 4:
            img_input = img_input[..., :3]
        img = Image.fromarray(
            (img_input * 255).astype(np.uint8) if img_input.max() <= 1 else img_input.astype(np.uint8)
        )
    else:
        raise ValueError("img_input must be path, bytes, or numpy array")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # (1, H, W, 3)
