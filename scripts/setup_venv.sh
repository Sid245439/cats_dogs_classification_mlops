#!/bin/bash
# Ensure venv exists and has MLflow + setuptools. Use this Python for all scripts.
# Run with: RECREATE_VENV=1 ./scripts/setup_venv.sh  to force fresh venv
set -e
cd "$(dirname "$0")/.."

if [ "$RECREATE_VENV" = "1" ] || [ ! -d "venv" ]; then
  echo "Creating fresh venv..."
  rm -rf venv
  python3 -m venv venv
fi

echo "Installing/upgrading pip..."
./venv/bin/pip install --upgrade pip

echo "Installing setuptools<75 (pkg_resources removed in setuptools 75+)..."
./venv/bin/pip install "setuptools>=65,<75"

echo "Verifying pkg_resources..."
./venv/bin/python -c "import pkg_resources; print('✓ pkg_resources OK')"

echo "Installing requirements..."
./venv/bin/pip install -r requirements.txt

echo "Verifying MLflow..."
./venv/bin/python -c "import mlflow; print('✓ MLflow OK')"

echo ""
echo "Done. Use: ./venv/bin/python"
