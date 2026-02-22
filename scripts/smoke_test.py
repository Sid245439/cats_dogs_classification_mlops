#!/usr/bin/env python3
"""Post-deploy smoke test: health + prediction. Fails pipeline if any check fails."""

import sys
import requests
from pathlib import Path

# Create a minimal test image (224x224 RGB)
import numpy as np
from PIL import Image
from io import BytesIO


def create_test_image():
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    # 1. Health
    r = requests.get(f"{base_url}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    assert data.get("status") == "healthy", f"Expected healthy, got {data}"

    # 2. Prediction
    img = create_test_image()
    r = requests.post(
        f"{base_url}/predict",
        files={"file": ("test.jpg", img, "image/jpeg")},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Prediction failed: {r.status_code} {r.text}")
    out = r.json()
    assert "label" in out or "probabilities" in out, f"Invalid response: {out}"
    print("Smoke tests passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Smoke test failed: {e}")
        sys.exit(1)
