#!/bin/bash
set -u

OUT_DIR="runs/diagnostic_v4"
LOG_FILE="/home/dev/diagnostic_v4.log"

echo "=== Starting diagnostic run at $(date -u) ===" | tee "$LOG_FILE"

cd /home/dev/h100-2

/home/dev/h100-2/.venv/bin/python -u scripts/train.py \
    --out "$OUT_DIR" \
    --cache /home/dev/cache \
    --overfit 16 \
    --steps 100 \
    --warmup 200 \
    --ac full \
    --min-std 0.0 \
    --max-std 0.40 \
    --preview-every 10,20,40,60,80,100 \
    --ckpt-every 20 \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "=== Training finished with exit code $EXIT_CODE at $(date -u) ===" >> "$LOG_FILE"
sync

echo "Calling givemeanode metadata self-stop endpoint..." >> "$LOG_FILE"
curl -sS -m 15 -X POST -H 'Metadata-Flavor: givemeanode' http://169.254.42.1/v1/stop >> "$LOG_FILE" 2>&1 || \
python3 -c "import urllib.request; req = urllib.request.Request('http://169.254.42.1/v1/stop', headers={'Metadata-Flavor': 'givemeanode'}, method='POST'); urllib.request.urlopen(req, timeout=15)" >> "$LOG_FILE" 2>&1 || true

echo "Shutdown request issued." >> "$LOG_FILE"
