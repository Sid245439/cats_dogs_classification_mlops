#!/bin/bash
# Full MLOps Demo: CI/CD trigger + Local Docker + Prediction
set -e
cd "$(dirname "$0")/.."

echo "=== 0. Setup venv (MLflow, setuptools) ==="
./scripts/setup_venv.sh
PYTHON=./venv/bin/python

echo ""
echo "=== 1. Trigger CI/CD ==="
if git config user.email >/dev/null 2>&1 && git config user.name >/dev/null 2>&1; then
  echo "<!-- $(date) -->" >> README.md
  git add . && git commit -m "Demo $(date +%Y-%m-%d)" && git push
else
  echo "Skipping git push (config not set). Run first:"
  echo "  git config --global user.email 'you@example.com'"
  echo "  git config --global user.name 'Your Name'"
fi

echo ""
echo "=== 2. Local Docker Demo ==="
$PYTHON scripts/download_data.py
$PYTHON scripts/prepare_data.py
$PYTHON -c "from src.training import train_and_track; train_and_track(epochs=2)"
docker build -t cats-dogs-mlops .
docker rm -f api 2>/dev/null || true
# Mount models from host (training runs first, so model exists)
docker run -d -p 8000:8000 -v "$(pwd)/models:/app/models:ro" --name api cats-dogs-mlops

echo ""
echo "Waiting for API..."
sleep 15

curl -s http://localhost:8000/health | $PYTHON -m json.tool
echo ""
$PYTHON scripts/smoke_test.py http://localhost:8000

echo ""
echo "Done! API at http://localhost:8000 | Stop: docker rm -f api"
