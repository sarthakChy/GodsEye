#!/usr/bin/env bash
set -e

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../env/bin/activate" ]; then
    source ../env/bin/activate
fi

python run_dashboard_v2.py "$@"
