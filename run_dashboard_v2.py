#!/usr/bin/env python
"""One-shot launcher for GodsEye Dashboard v2.

    python run_dashboard_v2.py                 # CPU, auto-detects weights
    python run_dashboard_v2.py --device cuda   # GPU
    python run_dashboard_v2.py --port 7862     # custom port
    python run_dashboard_v2.py --no-pipeline   # Instant launch without model (temporal tabs only)
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main():
    REPO = Path(__file__).resolve().parent
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    os.chdir(REPO)

    if "--device" not in sys.argv:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    APP = REPO / "temporal" / "viz" / "dashboard_v2.py"
    if not APP.exists():
        sys.exit(f"Dashboard v2 not found at {APP}")

    CKPT = REPO / "model" / "model.pth"
    DET_M = REPO / "checkpoints" / "detectors" / "yoloe-11m-seg-pf.pt"
    DET_S = REPO / "checkpoints" / "detectors" / "yoloe-11s-seg-pf.pt"
    DET = DET_M if DET_M.exists() else DET_S

    if "--no-pipeline" not in sys.argv:
        if "--ckpt" not in sys.argv and CKPT.exists():
            sys.argv += ["--ckpt", str(CKPT)]
        if "--det" not in sys.argv and DET.exists():
            sys.argv += ["--det", str(DET)]
        if "--device" not in sys.argv:
            sys.argv += ["--device", "cpu"]

    if "--port" not in sys.argv:
        sys.argv += ["--port", "7861"]

    print(f"[run_dashboard_v2] Launching GodsEye Dashboard v2 on port 7861...")
    print(f"[run_dashboard_v2] argv = {sys.argv[1:]}")
    runpy.run_path(str(APP), run_name="__main__")


if __name__ == "__main__":
    main()
