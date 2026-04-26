#!/usr/bin/env bash
# Optional: run from cron on your machine instead of (or in addition to) the GitHub Action.
# Appends a line to logs/heartbeat.log, commits, then you run: git push
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "$(date -u -Iseconds)Z heartbeat (local)" >> logs/heartbeat.log
git add logs/heartbeat.log
git diff --staged --quiet && { echo "Something to commit."; exit 0; }
git commit -m "chore: daily"
echo "Pushed? Run: git push"
