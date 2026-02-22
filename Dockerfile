FROM python:3.11-slim

WORKDIR /app

# Python deps (no apt packages – avoids Debian version conflicts)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY . /app

# Verify model exists (fail build if missing)
RUN test -f /app/models/model.h5 || test -f /app/models/model.keras || (echo "Model file missing in image" && exit 1)

# Ensure src is importable
ENV PYTHONPATH=/app

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
