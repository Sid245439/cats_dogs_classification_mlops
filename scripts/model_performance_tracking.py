#!/usr/bin/env python3
"""
M5: Model Performance Tracking (Post-Deployment)
Collects batch of requests with true labels and computes metrics.
"""

import sys
import json
import requests
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_synthetic_batch(n=20, seed=42):
    """Create synthetic images matching training data pattern (download_data create_sample_dataset)."""
    np.random.seed(seed)
    images, labels = [], []
    for i in range(n):
        label = i % 2  # alternate cat/dog (0=cat, 1=dog)
        base = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        if label == 0:  # cat - same as training: left 100 cols brighter
            base[:, :100] = np.clip(base[:, :100] + 30, 0, 255).astype(np.uint8)
        arr = np.clip(base + np.random.randn(224, 224, 3).astype(np.int16) * 5, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        images.append(buf.getvalue())
        labels.append(label)
    return images, labels


def load_real_test_data(data_dir="data/raw/cats_vs_dogs", max_per_class=20):
    """Load real test images if available. Returns (images, labels) or (None, None)."""
    from pathlib import Path
    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        return None, None
    images, labels = [], []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = 1 if "dog" in cls_dir.name.lower() else 0
        for fp in list(cls_dir.glob("*.jpg"))[:max_per_class] + list(cls_dir.glob("*.png"))[:5]:
            try:
                with open(fp, "rb") as f:
                    images.append(f.read())
                labels.append(label)
            except Exception:
                pass
    return (images, labels) if images else (None, None)


def evaluate_model(base_url="http://localhost:8000", images=None, labels=None):
    """Send batch to API, collect predictions, compute metrics."""
    if not images or not labels:
        images, labels = create_synthetic_batch(n=40)

    predictions = []
    for img_bytes in images:
        r = requests.post(
            f"{base_url}/predict",
            files={"file": ("img.jpg", img_bytes, "image/jpeg")},
            timeout=10,
        )
        if r.status_code != 200:
            predictions.append(-1)  # mark failure
            continue
        out = r.json()
        pred = 1 if out.get("label") == "dog" else 0
        predictions.append(pred)

    valid = [i for i, p in enumerate(predictions) if p >= 0]
    if not valid:
        print("No successful predictions")
        return {}

    y_true = [labels[i] for i in valid]
    y_pred = [predictions[i] for i in valid]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_samples": len(valid),
    }
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"Evaluating model at {base_url}...")
    images, labels = load_real_test_data()
    data_src = "real test set" if images else "synthetic (matches training pattern)"
    print(f"Using: {data_src}")
    metrics = evaluate_model(base_url, images, labels)
    print(json.dumps(metrics, indent=2))
    out_path = Path("logs/post_deploy_metrics.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
