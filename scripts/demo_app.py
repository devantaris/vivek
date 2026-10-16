#!/usr/bin/env python
"""Launch the PJ-03 Gradio demo.

Usage:
    python scripts/demo_app.py [--port 7860]

Trains the artifact first if it is missing (CPU, ~30s).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pj03.demo.app import main  # noqa: E402
from pj03.demo.predictor import DEFAULT_ARTIFACT  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PJ-03 demo app")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    artifact = REPO_ROOT / DEFAULT_ARTIFACT
    if not artifact.is_file():
        print("Demo artifact missing - training it now (CPU, ~30s) ...")
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "train_demo_model.py")],
            check=True,
        )
    main(artifact, args.port)
