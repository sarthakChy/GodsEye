#!/usr/bin/env python
"""One-shot launcher for GodsEye Dashboard v2.

    python run_dashboard_v2.py                 # auto GPU (falls back to CPU if CUDA unavailable)
    python run_dashboard_v2.py --device cuda   # force GPU
    python run_dashboard_v2.py --device cpu    # force CPU
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

    # Remove CUDA masking so the GPU is accessible by default
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    APP = REPO / "temporal" / "viz" / "dashboard_v2.py"
    if not APP.exists():
        sys.exit(f"Dashboard v2 not found at {APP}")

    CKPT = REPO / "model" / "model.pth"
    DET_M = REPO / "checkpoints" / "detectors" / "yoloe-11m-seg-pf.pt"
    DET_S = REPO / "checkpoints" / "detectors" / "yoloe-11s-seg-pf.pt"

    if "--no-pipeline" not in sys.argv:
        # Auto-download missing model weights or detectors if not present
        if not CKPT.exists() or (not DET_M.exists() and not DET_S.exists()):
            print("[run_dashboard_v2] Model weights or detectors not found. Running auto-download...")
            try:
                from setup_env import ensure_model_weights
                ensure_model_weights()
            except Exception as e:
                print(f"[run_dashboard_v2] Auto-download failed: {e}. Run 'python setup_env.py' to configure.")

        DET = DET_M if DET_M.exists() else DET_S
        if "--ckpt" not in sys.argv and CKPT.exists():
            sys.argv += ["--ckpt", str(CKPT)]
        if "--det" not in sys.argv and DET.exists():
            sys.argv += ["--det", str(DET)]
        # Default to CUDA; dashboard_v2.py auto-falls back to CPU if CUDA is unavailable
        if "--device" not in sys.argv:
            sys.argv += ["--device", "cuda"]

    if "--port" not in sys.argv:
        sys.argv += ["--port", "7861"]

    print(f"[run_dashboard_v2] Launching GodsEye Dashboard v2 on port 7861...")
    print(f"[run_dashboard_v2] argv = {sys.argv[1:]}")
    runpy.run_path(str(APP), run_name="__main__")


if __name__ == "__main__":
    main()
