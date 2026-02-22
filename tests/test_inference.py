"""Unit tests for inference/model utilities."""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_preprocess_for_inference_produces_valid_input():
    """preprocess output shape matches CNN input (1, 224, 224, 3)."""
    from src.preprocessing import preprocess_for_inference
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out = preprocess_for_inference(img)
    assert out.ndim == 4
    assert out.shape[0] == 1
    assert out.shape[1:3] == (224, 224)
    assert out.shape[3] == 3


def test_preprocess_for_inference_bytes():
    """preprocess_for_inference accepts image bytes."""
    from src.preprocessing import preprocess_for_inference
    from PIL import Image
    from io import BytesIO
    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    out = preprocess_for_inference(buf.getvalue())
    assert out.shape == (1, 224, 224, 3)
