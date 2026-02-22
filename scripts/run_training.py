#!/usr/bin/env python3
"""Run full training pipeline: download -> prepare -> train."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    # 1. Download
    subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py")], check=True, cwd=str(ROOT))

    # 2. Prepare
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py")], check=True, cwd=str(ROOT))

    # 3. Train
    import src.training
    src.training.train_and_track(epochs=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
