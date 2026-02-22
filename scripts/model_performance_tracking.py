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
    """Create synthetic images + labels for testing (simulated requests)."""
    np.random.seed(seed)
    images, labels = [], []
    for i in range(n):
        label = i % 2  # alternate cat/dog
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        if label == 0:  # cat
            arr[:, :50] = np.clip(arr[:, :50] + 20, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        images.append(buf.getvalue())
        labels.append(label)
    return images, labels


def evaluate_model(base_url="http://localhost:8000", images=None, labels=None):
    """Send batch to API, collect predictions, compute metrics."""
    if images is None or labels is None:
        images, labels = create_synthetic_batch()

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
    metrics = evaluate_model(base_url)
    print(json.dumps(metrics, indent=2))
    out_path = Path("logs/post_deploy_metrics.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
