#!/usr/bin/env python3
"""Prepare and save preprocessed data to data/processed for DVC tracking."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_dataset
import numpy as np

def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    np.savez_compressed(
        "data/processed/dataset.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print("✓ Saved data/processed/dataset.npz")
    return 0

if __name__ == "__main__":
    sys.exit(main())
