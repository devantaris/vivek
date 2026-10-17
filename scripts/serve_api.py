#!/usr/bin/env python
"""Launch the PJ-03 REST API.

Usage:
    python scripts/serve_api.py [--host 0.0.0.0] [--port 8000]

Trains the demo artifact first if missing (CPU, ~30s).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from pj03.demo.predictor import DEFAULT_ARTIFACT, DemoPredictor

    artifact = REPO_ROOT / DEFAULT_ARTIFACT
    if not artifact.is_file():
        print("Demo artifact missing - training it now (CPU, ~30s) ...")
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "train_demo_model.py")],
            check=True,
        )
    predictor = DemoPredictor(artifact)

    import uvicorn

    from pj03.service import create_app

    app = create_app(predictor, decision_log=REPO_ROOT / "logs" / "decisions.jsonl")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
