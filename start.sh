#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[ForgeShield] Starting Python detector API on port 8000..."
cd "$ROOT/detector"
python detector_api.py &
DETECTOR_PID=$!

echo "[ForgeShield] Waiting for Python API to initialise..."
sleep 6

echo "[ForgeShield] Starting Node.js backend on port 5000..."
cd "$ROOT/backend"
npm start &
BACKEND_PID=$!

echo ""
echo "[ForgeShield] Both services started."
echo "  Backend  : http://localhost:5000/api/health  (PID $BACKEND_PID)"
echo "  Detector : http://localhost:8000/health      (PID $DETECTOR_PID)"
echo ""
echo "Press Ctrl+C to stop both services."

trap "kill $DETECTOR_PID $BACKEND_PID 2>/dev/null; exit" INT TERM
wait
