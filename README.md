# MLOps Assignment 2: Cats vs Dogs Binary Classification

End-to-end MLOps pipeline for a pet adoption platform.

## Project Structure

```
ML_OPS_Assignment_2/
├── app.py                 # FastAPI inference service
├── src/
│   ├── preprocessing.py   # Image preprocessing, augmentation
│   └── training.py        # CNN training with MLflow
├── scripts/
│   ├── download_data.py   # Kaggle/sample dataset download
│   ├── prepare_data.py    # Preprocess & save for DVC
│   ├── run_training.py     # Full pipeline
│   ├── smoke_test.py      # Post-deploy smoke tests
│   └── model_performance_tracking.py  # M5 metrics
├── tests/
│   ├── test_preprocessing.py
│   └── test_inference.py
├── k8s/deployment.yaml    # Kubernetes Deployment + Service
├── monitoring/prometheus.yml
├── dvc.yaml               # DVC pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci-cd.yml
```

## Quick Start

### 1. Data & Training

```bash
# Create venv
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Download data (Kaggle or sample)
python scripts/download_data.py

# Prepare (80/10/10 split, 224x224)
python scripts/prepare_data.py

# Train (MLflow tracks runs)
python -c "from src.training import train_and_track; train_and_track(epochs=5)"
```

### 2. Run Inference API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Swagger UI**: `GET /docs` – interactive API docs for testing POST /predict
- Health: `GET /health`
- Predict: `POST /predict` (multipart image file)
- Metrics: `GET /metrics` (Prometheus)

### Prometheus

1. **Start the API** (uvicorn or Docker) on port 8000.
2. **Run Prometheus:**
   ```bash
   docker run -d -p 9090:9090 \
     -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
     prom/prometheus --config.file=/etc/prometheus/prometheus.yml
   ```
3. **Open** http://localhost:9090 → Status → Targets (verify scrape).
4. **Query** e.g. `cats_dogs_api_requests_total` or `rate(cats_dogs_api_request_latency_seconds_sum[1m])`.

   **Linux:** In `prometheus.yml`, use `172.17.0.1:8000` instead of `host.docker.internal:8000`.

   **Without Docker:** `brew install prometheus` (macOS), then `prometheus --config.file=monitoring/prometheus-host.yml`.

   **No data?** See `scripts/prometheus_troubleshoot.md`. Quick checks: `curl localhost:8000/metrics`, then http://localhost:9090/targets (target must be UP).

### Stress test (for Prometheus metrics)

```bash
# Ensure API + Prometheus are running, then:
python scripts/stress_test.py http://localhost:8000 200 20
# 200 requests, 20 concurrent – check Prometheus for cats_dogs_api_requests_total
```

### 3. Docker

```bash
# Build & run
docker build -t cats-dogs-mlops .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro cats-dogs-mlops

# Or with Compose
docker compose up -d
```

### 4. Smoke Test

```bash
python scripts/smoke_test.py http://localhost:8000
```

### 5. Model Performance Tracking (M5)

```bash
python scripts/model_performance_tracking.py http://localhost:8000
```

## CI/CD

- **CI**: On push/PR: tests, train, build image, push to GHCR
- **CD**: On main: deploy image, smoke tests

## DVC

```bash
dvc init
dvc add data/processed/dataset.npz
dvc add models/model.h5
git add .dvc data/processed.dvc models.dvc
```

## Deliverables Checklist

- [x] Git versioning
- [x] DVC/data versioning (dvc.yaml, .dvc/config)
- [x] Baseline CNN, MLflow tracking
- [x] FastAPI /health, /predict
- [x] requirements.txt (pinned)
- [x] Dockerfile, docker-compose
- [x] Unit tests (pytest)
- [x] CI (GitHub Actions)
- [x] CD (deploy on main), smoke tests
- [x] Logging, Prometheus metrics
- [x] Post-deploy performance tracking
<!-- Sun Feb 22 16:17:17 IST 2026 -->
<!-- Sun Feb 22 16:19:54 IST 2026 -->
<!-- Sun Feb 22 16:20:14 IST 2026 -->
