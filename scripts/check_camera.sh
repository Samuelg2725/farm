#!/usr/bin/env bash
# Checks the camera is reachable and both video streams work (macOS + Linux).
# Usage: ./scripts/check_camera.sh
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "No .env – run: cp .env.example .env  and edit it"; exit 1; }
set -a; . ./.env; set +a

echo "== Ping $CAM_IP"
if ping -c 2 "$CAM_IP" >/dev/null 2>&1; then echo "OK"; else echo "FAIL – camera not reachable. Check cable/switch and the IP in .env"; exit 1; fi

for port in 80 554 8000; do
  if nc -z -w 3 "$CAM_IP" "$port" >/dev/null 2>&1; then echo "Port $port open"
  else echo "Port $port CLOSED – enable it in the camera's network/server settings (554=RTSP, 8000=ONVIF, 80=HTTP)"; fi
done

if ! command -v ffprobe >/dev/null; then
  echo "ffprobe not installed – on Mac: brew install ffmpeg   (skipping stream test)"; exit 0
fi
for s in main sub; do
  echo "== Stream $s"
  ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate -of compact=p=0 \
    "rtsp://$CAM_USER:$CAM_PASS@$CAM_IP:554/Preview_01_$s" || echo "FAIL – check user/password and that RTSP is enabled"
done
