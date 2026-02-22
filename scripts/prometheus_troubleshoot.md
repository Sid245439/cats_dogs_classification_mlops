# Prometheus "No Data" – Troubleshooting

## 1. Verify API /metrics returns data
```bash
curl http://localhost:8000/metrics
```
You should see `cats_dogs_api_requests_total`, `cats_dogs_api_request_latency_seconds`, etc.

## 2. Make some requests first
Metrics appear after requests. Run:
```bash
curl http://localhost:8000/health
python scripts/stress_test.py http://localhost:8000 20 5
```
Then check /metrics again.

## 3. Prometheus can't reach API (Docker)

**macOS/Windows:** Use `host.docker.internal:8000` in prometheus.yml.
**Linux:** Run Prometheus with host network so it can see localhost:
```bash
docker run -d -p 9090:9090 --network=host \
  -v $(pwd)/monitoring/prometheus-host.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus --config.file=/etc/prometheus/prometheus.yml
```
Or add host gateway:
```bash
docker run -d -p 9090:9090 --add-host=host.docker.internal:host-gateway \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus --config.file=/etc/prometheus/prometheus.yml
```

## 4. Check Prometheus targets
Open http://localhost:9090/targets – the cats-dogs-api job should show **UP** (green).

## 5. Queries that should work
- `up` – shows if targets are reachable (1 = up)
- `cats_dogs_api_requests_total` – after at least one API request
- `process_resident_memory_bytes` – Python process metrics (always present)
