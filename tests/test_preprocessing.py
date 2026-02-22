"""Unit tests for preprocessing functions."""

import tempfile
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    load_image,
    augment_image,
    preprocess_for_inference,
    IMG_SIZE,
    CLASSES,
)


def test_load_image_valid_file():
    """load_image loads valid image and returns correct shape."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(f.name)
    try:
        out = load_image(f.name)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.float32
        assert 0 <= out.min() <= out.max() <= 1
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_load_image_requires_valid_path():
    """load_image should raise for invalid path."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_image("/nonexistent/path/image.jpg")


def test_augment_image_returns_same_shape():
    """augment_image should preserve input shape."""
    np.random.seed(42)
    img = np.random.rand(224, 224, 3).astype(np.float32)
    out = augment_image(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_preprocess_for_inference_numpy():
    """preprocess_for_inference accepts numpy array."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    out = preprocess_for_inference(img)
    assert out.shape == (1, 224, 224, 3)
    assert out.dtype == np.float32
    assert 0 <= out.min() <= out.max() <= 1


def test_preprocess_for_inference_grayscale():
    """preprocess_for_inference converts 2D to 3 channel."""
    img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    out = preprocess_for_inference(img)
    assert out.shape == (1, 224, 224, 3)


def test_classes_defined():
    """CLASSES should be cat and dog."""
    assert CLASSES == ["cat", "dog"]
    assert len(CLASSES) == 2
