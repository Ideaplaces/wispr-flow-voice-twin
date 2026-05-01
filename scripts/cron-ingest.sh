#!/usr/bin/env bash
# Cron entry: pick up any wispr-flow-delta.jsonl in the inbox and run the
# incremental ingest. No-op when the inbox is empty, so it's safe to fire
# every 15 minutes regardless of when Mac last rsynced.
set -euo pipefail

REPO="/home/chipdev/ideaplaces-meta/wispr-flow-voice-twin"
INBOX="$REPO/data/inbox/wispr-flow-delta.jsonl"
LOG_DIR="$REPO/data/logs"
LOG_FILE="$LOG_DIR/ingest.log"

mkdir -p "$LOG_DIR"

# Skip silently if there's nothing to ingest. Cron noise reduction.
if [ ! -f "$INBOX" ]; then
    exit 0
fi

cd "$REPO"
{
    echo
    echo "===== $(date -u +%FT%TZ) ingest ====="
    .venv/bin/python pipeline/07_ingest_delta.py
} >> "$LOG_FILE" 2>&1
