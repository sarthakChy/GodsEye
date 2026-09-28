#!/usr/bin/env python
"""One-shot launcher for the GodsEye styled dashboard.

    python run_dashboard.py                 # CPU, model/model.pth, yoloe-11m detector
    python run_dashboard.py --device cuda   # GPU
    python run_dashboard.py --port 7861     # custom port
    python run_dashboard.py --share         # public gradio link
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.chdir(REPO)

if "--device" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

APP = REPO / "deploy" / "gradio_app_styled.py"
if not APP.exists():
    sys.exit(f"styled dashboard not found at {APP}")

CKPT = REPO / "model" / "model.pth"
DET  = REPO / "checkpoints" / "detectors" / "yoloe-11m-seg-pf.pt"

if not CKPT.exists():
    sys.exit(
        f"checkpoint missing: {CKPT}\n"
        "Download it once with:\n"
        "  python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('maelic/relsgg-vits16plus', local_dir='model')\""
    )

if "--ckpt" not in sys.argv:
    sys.argv += ["--ckpt", str(CKPT)]
if "--det" not in sys.argv and DET.exists():
    sys.argv += ["--det", str(DET)]
if "--device" not in sys.argv:
    sys.argv += ["--device", "cpu"]

print(f"[run_dashboard] cwd = {Path.cwd()}")
print(f"[run_dashboard] argv = {sys.argv[1:]}")
runpy.run_path(str(APP), run_name="__main__")
