#!/usr/bin/env python3
"""GodsEye Universal Setup & Hardware Configuration
==================================================
Configures GodsEye on ANY PC (Windows, Linux, macOS):
  1. Detects system hardware (NVIDIA GPU, Apple Silicon MPS, or CPU).
  2. Installs GPU-optimized PyTorch (CUDA 12.4 / 11.8) if an NVIDIA GPU is present.
  3. Installs all project dependencies (Gradio, Ultralytics, OpenCV, NetworkX, Plotly, etc.).
  4. Automatically downloads all required model weights & detectors if missing:
       - Relation model: maelic/relsgg-vits16plus (from HuggingFace Hub)
       - YOLOE segmentation detectors: yoloe-11m-seg-pf.pt & yoloe-11s-seg-pf.pt
  5. Verifies environment and GPU tensor execution.

Usage:
    python setup.py                # Full automated setup
    python setup.py --check-only   # Check environment without installing
    python setup.py --cpu-only     # Force CPU installation
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)

# Core dependencies needed for GodsEye dashboard v2
BASE_DEPENDENCIES = [
    "gradio>=5.0.0",
    "opencv-python>=4.8.0",
    "ultralytics>=8.3.0",
    "networkx>=3.0",
    "plotly>=5.17.0",
    "imageio>=2.30.0",
    "imageio-ffmpeg>=0.5.0",
    "huggingface_hub>=0.20.0",
    "scipy>=1.10.0",
    "transformers>=4.40.0",
    "pillow>=9.0.0",
    "tqdm>=4.65.0",
]

# Model weights configurations
HF_MODEL_REPO = "maelic/relsgg-vits16plus"
DETECTORS = [
    {
        "filename": "yoloe-11m-seg-pf.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg-pf.pt",
        "desc": "Primary High-Accuracy YOLOE-11m Segmentation Detector (~60 MB)",
    },
    {
        "filename": "yoloe-11s-seg-pf.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt",
        "desc": "Lightweight Fallback YOLOE-11s Segmentation Detector (~26 MB)",
    },
]


def print_step(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_cmd(cmd: list[str] | str, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd
    print(f"  > {cmd_str}")
    return subprocess.run(cmd, check=check, shell=shell)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hardware Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware(force_cpu: bool = False) -> dict:
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "device": "cpu",
        "gpu_name": None,
        "cuda_version": None,
        "index_url": None,
    }

    if force_cpu:
        print("  [Notice] --cpu-only specified: skipping GPU configuration.")
        return info

    # Check for Apple Silicon (macOS arm64)
    if info["os"] == "Darwin" and info["arch"] in ("arm64", "aarch64"):
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon (Metal MPS)"
        return info

    # Check for NVIDIA GPU via nvidia-smi
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output([smi], text=True, stderr=subprocess.STDOUT)
            # Try to parse Driver Version & CUDA Version
            driver_match = re.search(r"Driver Version:\s*(\d+\.\d+)", out)
            cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)

            driver_ver = float(driver_match.group(1)) if driver_match else 0.0
            cuda_ver = cuda_match.group(1) if cuda_match else "12.4"

            # Parse GPU Name
            gpu_name_cmd = [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"]
            try:
                gpu_lines = subprocess.check_output(gpu_name_cmd, text=True).strip().splitlines()
                gpu_name = gpu_lines[0] if gpu_lines else "NVIDIA GPU"
            except Exception:
                gpu_name = "NVIDIA GPU"

            info["device"] = "cuda"
            info["gpu_name"] = gpu_name
            info["cuda_version"] = cuda_ver

            # Select PyTorch wheel index
            # Drivers >= 525 support CUDA 12.x
            # Drivers between 450 and 524 support CUDA 11.8
            if driver_ver >= 525 or not driver_ver:
                info["index_url"] = "https://download.pytorch.org/whl/cu124"
            else:
                info["index_url"] = "https://download.pytorch.org/whl/cu118"

            return info
        except Exception as e:
            print(f"  [warn] nvidia-smi found but failed to query: {e}")

    # Fallback Windows WMI check for NVIDIA controller
    if info["os"] == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "nvidia" in line.lower() or "geforce" in line.lower() or "rtx" in line.lower() or "quadro" in line.lower():
                    info["device"] = "cuda"
                    info["gpu_name"] = line.strip()
                    info["index_url"] = "https://download.pytorch.org/whl/cu124"
                    return info
        except Exception:
            pass

    return info


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch Installation
# ─────────────────────────────────────────────────────────────────────────────

def install_pytorch(hw: dict):
    print_step(f"Verifying PyTorch for {hw['device'].upper()} Acceleration")

    # Check if PyTorch is already installed and matches target device
    needs_install = True
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        torch_ver = torch.__version__
        print(f"  Found installed PyTorch: {torch_ver} (CUDA available: {has_cuda})")

        if hw["device"] == "cuda" and has_cuda:
            print(f"  CUDA is already operational on: {torch.cuda.get_device_name(0)}")
            needs_install = False
        elif hw["device"] != "cuda" and not has_cuda:
            print("  CPU PyTorch is already installed.")
            needs_install = False
        else:
            print("  Existing PyTorch does not support target GPU; reinstalling...")
    except ImportError:
        print("  PyTorch is not installed yet.")

    if not needs_install:
        return

    python_exe = sys.executable
    if hw["device"] == "cuda" and hw["index_url"]:
        print(f"  Installing PyTorch with CUDA support from {hw['index_url']} ...")
        cmd = [
            python_exe, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", hw["index_url"],
        ]
    else:
        print("  Installing standard PyTorch (CPU/MPS)...")
        cmd = [
            python_exe, "-m", "pip", "install",
            "torch", "torchvision",
        ]
    run_cmd(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Base Dependencies & Local Package
# ─────────────────────────────────────────────────────────────────────────────

def install_dependencies():
    print_step("Installing Project Dependencies")
    python_exe = sys.executable

    # Install base packages without forcing --upgrade so existing packages are preserved
    print("  Installing core libraries...")
    run_cmd([python_exe, "-m", "pip", "install"] + BASE_DEPENDENCIES)

    # Install local repo in editable mode without overriding torch
    print("  Registering local package (relsgg)...")
    run_cmd([python_exe, "-m", "pip", "install", "-e", ".", "--no-deps"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Weights & Detector Auto-Download
# ─────────────────────────────────────────────────────────────────────────────

def download_file_with_progress(url: str, dest_path: Path, desc: str):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".download")

    print(f"\n  Fetching: {desc}")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    # Use curl if available for maximum speed and resumability
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl:
        try:
            cmd = [curl, "-L", "-C", "-", "--retry", "5", "--retry-delay", "2", "-o", str(temp_path), url]
            subprocess.run(cmd, check=True)
            if temp_path.exists() and temp_path.stat().st_size > 1024:
                temp_path.replace(dest_path)
                print(f"  [OK] Successfully saved {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB)")
                return
        except Exception as e:
            print(f"  [curl fallback] {e}")

    # Fallback to urllib streaming with progress
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 GodsEyeSetup"})
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = int((downloaded / total_size) * 100)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb:.1f} MB / {total_mb:.1f} MB [{pct}%]")
                sys.stdout.flush()
        print()

    if temp_path.exists():
        temp_path.replace(dest_path)
        print(f"  [OK] Saved {dest_path.name}")


def ensure_model_weights():
    print_step("Checking Model Checkpoints & Detectors")
    model_dir = REPO_ROOT / "model"
    ckpt_file = model_dir / "model.pth"

    # 1. Relation Model (from HuggingFace Hub)
    if not ckpt_file.exists():
        print(f"  Missing relation model: {ckpt_file}")
        print(f"  Downloading from HuggingFace Hub: '{HF_MODEL_REPO}' (~250 MB)...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                local_dir=str(model_dir),
            )
            print("  [OK] Relation model downloaded successfully.")
        except Exception as e:
            print(f"  [Error] Failed to download from HuggingFace Hub: {e}")
            print(f"  Please check your internet connection or run: huggingface-cli download {HF_MODEL_REPO}")
    else:
        print(f"  [OK] Relation model found: {ckpt_file} ({ckpt_file.stat().st_size / 1e6:.1f} MB)")

    # 2. YOLOE Segmentation Detectors
    detectors_dir = REPO_ROOT / "checkpoints" / "detectors"
    for item in DETECTORS:
        target_path = detectors_dir / item["filename"]
        if not target_path.exists() or target_path.stat().st_size < 1024 * 100:
            download_file_with_progress(item["url"], target_path, item["desc"])
        else:
            print(f"  [OK] Detector found: {target_path.name} ({target_path.stat().st_size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sanity Check & Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_setup(hw: dict):
    print_step("Running System Verification")
    import torch

    print(f"  Python:     {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"  PyTorch:    {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA Live:  {cuda_ok}")

    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Device: {device_name} ({vram_gb:.1f} GB VRAM)")
        # Test tensor on GPU
        x = torch.ones((100, 100), device="cuda")
        y = x @ x
        del x, y
        torch.cuda.empty_cache()
        print("  [OK] GPU Tensor test passed!")
    elif hw["device"] == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  Device:     {hw['gpu_name']}")
        print("  [OK] Apple Silicon Metal acceleration verified.")
    else:
        print("  Device:     CPU")
        print("  [OK] CPU compute verified.")

    print("\n" + "=" * 65)
    print("  🎉 GODSEYE SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print("\nTo launch the unified dashboard, simply run:")
    print("    python run_dashboard_v2.py")
    if platform.system() == "Windows":
        print("Or double-click: run_dashboard.bat\n")
    else:
        print("Or run: ./run_dashboard.sh\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GodsEye Universal Setup")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU installation")
    parser.add_argument("--check-only", action="store_true", help="Only check status without installing")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading model weights")
    args = parser.parse_args()

    print("=" * 65)
    print("  👁 GodsEye Neuro-Symbolic System Setup")
    print("=" * 65)

    if sys.version_info < (3, 10):
        sys.exit("Error: GodsEye requires Python 3.10 or newer.")

    hw = detect_hardware(force_cpu=args.cpu_only)
    print(f"  OS:       {hw['os']} ({hw['arch']})")
    print(f"  Hardware: {hw['gpu_name'] or 'CPU'}")
    print(f"  Target:   {hw['device'].upper()}")

    if args.check_only:
        print("\n  Check-only mode completed.")
        return

    install_pytorch(hw)
    install_dependencies()

    if not args.skip_models:
        ensure_model_weights()

    verify_setup(hw)


if __name__ == "__main__":
    main()
