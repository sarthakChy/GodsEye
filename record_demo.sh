#!/usr/bin/env bash
# Records a 60-second demo of the GodsEye dashboard to godseye_demo.mp4
# Usage: bash record_demo.sh [duration_seconds]
set -euo pipefail

DURATION="${1:-60}"
OUT="godseye_demo.mp4"

echo "=== GodsEye demo recorder ==="
echo "Duration: ${DURATION}s"
echo "Output:   ${OUT}"

# Check DISPLAY
if [ -z "${DISPLAY:-}" ]; then
  echo "ERROR: DISPLAY is not set. This script needs a graphical session."
  exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg >/dev/null; then
  echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
  exit 1
fi

# Launch the dashboard in the background
echo "[1/3] Launching dashboard…"
python -m temporal.viz.dashboard_ui >/tmp/godseye_dash.log 2>&1 &
DASH_PID=$!

# Wait for the port to open
echo "[2/3] Waiting for http://127.0.0.1:7860 …"
for i in $(seq 1 30); do
  if curl -s -o /dev/null "http://127.0.0.1:7860"; then
    echo "      dashboard is up"
    break
  fi
  sleep 1
done

# Give the browser a moment to be opened by the user
echo
echo ">>> Open http://127.0.0.1:7860 in a browser now."
echo ">>> Recording will start in 5 seconds…"
sleep 5

# Record the screen
echo "[3/3] Recording for ${DURATION}s …"
ffmpeg -y -f x11grab -framerate 30 -video_size 1920x1080 -i "${DISPLAY}" \
       -t "${DURATION}" -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p \
       "${OUT}"

# Clean up
kill "${DASH_PID}" 2>/dev/null || true
wait "${DASH_PID}" 2>/dev/null || true

echo
echo "Done. Saved to ${OUT}"
echo "Play it with:  xdg-open ${OUT}"
