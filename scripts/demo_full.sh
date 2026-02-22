#!/bin/bash
# Full MLOps Demo: CI/CD trigger + Local Docker + Prediction
set -e
cd "$(dirname "$0")/.."

echo "=== 1. Trigger CI/CD ==="
echo "<!-- $(date) -->" >> README.md
git add . && git commit -m "Demo $(date +%Y-%m-%d)" && git push

echo ""
echo "=== 2. Local Docker Demo ==="
python scripts/download_data.py
python scripts/prepare_data.py
python -c "from src.training import train_and_track; train_and_track(epochs=2)"
docker build -t cats-dogs-mlops .
docker rm -f api 2>/dev/null || true
docker run -d -p 8000:8000 -v "$(pwd)/models:/app/models:ro" --name api cats-dogs-mlops

echo ""
echo "Waiting for API..."
sleep 15

curl -s http://localhost:8000/health | python -m json.tool
echo ""
python scripts/smoke_test.py http://localhost:8000

echo ""
echo "Done! API at http://localhost:8000 | Stop: docker rm -f api"
