#!/usr/bin/env bash
# =================================================================
#   GodsEye Setup - Hardware Detection & Auto-Installation (Linux/macOS)
# =================================================================
set -e

echo "================================================================="
echo "  GodsEye Universal Setup (Linux / macOS)"
echo "================================================================="

# 1. Detect Python
PYTHON_BIN=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3.10+ was not found!"
    echo "Please install Python 3.10, 3.11, or 3.12 before running setup."
    exit 1
fi

echo "[*] Using Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# 2. Virtual Environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ ! -d ".venv" ]; then
        echo "[*] Creating virtual environment (.venv)..."
        "$PYTHON_BIN" -m venv .venv
    fi
    if [ -f ".venv/bin/activate" ]; then
        echo "[*] Activating .venv..."
        source .venv/bin/activate
    fi
fi

# 3. Run Setup Engine
python setup_env.py "$@"

echo ""
echo "================================================================="
echo "  Setup complete! Launch the dashboard with:"
echo "    ./run_dashboard.sh   (or: python run_dashboard_v2.py)"
echo "================================================================="
