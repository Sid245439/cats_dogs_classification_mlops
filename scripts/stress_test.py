#!/usr/bin/env python3
"""
Stress test the API to generate metrics for Prometheus.
Usage: python scripts/stress_test.py [base_url] [num_requests] [concurrency]
"""
import sys
import concurrent.futures
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_test_image():
    """Create a minimal 224x224 RGB test image."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def single_predict_request(base_url):
    """Make one POST /predict request."""
    try:
        img = create_test_image()
        r = requests.post(
            f"{base_url}/predict",
            files={"file": ("test.jpg", img, "image/jpeg")},
            timeout=10,
        )
        r.raise_for_status()
        return 1
    except Exception:
        return 0


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print(f"Stress test POST /predict: {base_url} | {num_requests} requests | {concurrency} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_predict_request, base_url) for _ in range(num_requests)]
        success = sum(f.result() for f in concurrent.futures.as_completed(futures))

    print(f"Done: {success}/{num_requests} successful")
    print("Check Prometheus: cats_dogs_api_requests_total, cats_dogs_api_request_latency_seconds")


if __name__ == "__main__":
    main()
