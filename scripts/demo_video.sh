#!/bin/bash
#
# MLOps Demo Video Script
# Run this while recording. Shows: local train → Docker → prediction
# (CI/CD part: show in browser - GitHub Actions tab)
#

set -e
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

echo "=========================================="
echo "MLOps Demo - Cats vs Dogs Pipeline"
echo "=========================================="

# Step 1: Ensure data & model exist
echo ""
echo "[1/5] Downloading data..."
python scripts/download_data.py

echo ""
echo "[2/5] Preparing data..."
python scripts/prepare_data.py

echo ""
echo "[3/5] Training model (2 epochs for demo)..."
python -c "from src.training import train_and_track; train_and_track(epochs=2)"

# Step 2: Build Docker image
echo ""
echo "[4/5] Building Docker image..."
docker build -t cats-dogs-mlops .

# Step 3: Run container
echo ""
echo "[5/5] Starting API container..."
docker rm -f cats-dogs-api 2>/dev/null || true
docker run -d -p 8000:8000 -v "${PROJECT_DIR}/models:/app/models:ro" --name cats-dogs-api cats-dogs-mlops

echo ""
echo "Waiting for API to start..."
sleep 20

# Step 4: Smoke tests
echo ""
echo "--- Health Check ---"
curl -s http://localhost:8000/health | python -m json.tool

echo ""
echo "--- Smoke Test ---"
python scripts/smoke_test.py http://localhost:8000

echo ""
echo "--- Metrics ---"
curl -s http://localhost:8000/metrics | head -20

echo ""
echo "=========================================="
echo "Demo complete! API running at http://localhost:8000"
echo "Try: curl -X POST -F 'file=@image.jpg' http://localhost:8000/predict"
echo ""
echo "To stop: docker rm -f cats-dogs-api"
echo "=========================================="
