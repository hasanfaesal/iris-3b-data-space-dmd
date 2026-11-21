#!/bin/bash
set -u

# Stage 9 main run launcher for node iris-dsdmd-h100-s7.
# Intended to be started detached (run_command detach:true); all trainer
# flags below are repo defaults except --out/--cache/--ac. Deadline comes
# from the credit expiry default (2026-10-26T06:53Z).

cd /home/dev/h100-2
mkdir -p logs
LOG_FILE=logs/main.log

echo "=== Starting main run at $(date -u) ===" >> "$LOG_FILE"

.venv/bin/python -u scripts/train.py \
    --out runs/main \
    --cache /home/dev/cache \
    --ac full \
    >> "$LOG_FILE" 2>&1
echo "exit $? at $(date -u)" >> "$LOG_FILE"

curl -sS -m 15 -X POST -H 'Metadata-Flavor: givemeanode' http://169.254.42.1/v1/stop
echo "Shutdown request issued." >> "$LOG_FILE"
