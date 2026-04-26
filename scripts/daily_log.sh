#!/usr/bin/env bash
# Append one line to logs/daily.md: date, git tip, optional note.
# Usage: ./scripts/daily_log.sh
#        ./scripts/daily_log.sh "ran eval, need to plot ROC"
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/daily.md"
mkdir -p "$LOG_DIR"
if [[ ! -f "$LOG" ]]; then
  echo "# Daily log" > "$LOG"
  echo "" >> "$LOG"
  echo "Run \`./scripts/daily_log.sh\` with an optional message; commit this file when you want history on GitHub." >> "$LOG"
  echo "" >> "$LOG"
fi
DAY=$(date +%Y-%m-%d)
TIME=$(date '+%H:%M')
cd "$ROOT" || exit 1
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
LINE=$(git log -1 --oneline 2>/dev/null || echo "not a git repo or no commits")
BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
MSG="${*:-}"
if [[ -n "$MSG" ]]; then
  echo "- **$DAY** $TIME · \`$SHA\` $BR — $LINE — *$MSG*" >> "$LOG"
else
  echo "- **$DAY** $TIME · \`$SHA\` $BR — $LINE" >> "$LOG"
fi
echo "Appended: $LOG"
