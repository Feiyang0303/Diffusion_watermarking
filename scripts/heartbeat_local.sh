#!/usr/bin/env bash
# Optional: run from cron on your machine instead of (or in addition to) the GitHub Action.
# Appends a line to logs/heartbeat.log, commits, then you run: git push
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
echo "$(date -u -Iseconds)Z heartbeat (local)" >> logs/heartbeat.log
git add logs/heartbeat.log
if git diff --staged --quiet; then
  echo "Nothing new to commit."
  exit 0
fi
git commit -m "chore: daily"
echo "Committed. Run: git push"
