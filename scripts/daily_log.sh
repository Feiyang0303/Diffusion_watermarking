#!/usr/bin/env bash
# Append one line to logs/daily.md: date, git tip, optional note.
# Usage: ./scripts/daily_log.sh
#        ./scripts/daily_log.sh "ran eval, need to plot ROC"
#        ./scripts/daily_log.sh --random-dated-commits [N] ["optional note"]
#          N commits (default: random integer in 2..6), each with a random
#          author/committer date in the past DAILY_LOG_RANDOM_DAYS days (default 365)
#          and one new line in daily.md; run from repo root with git configured.
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

cd "$ROOT" || exit 1
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
LINE=$(git log -1 --oneline 2>/dev/null || echo "not a git repo or no commits")
BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")

_random_line_and_git_date() {
  # Prints: DAY<TAB>TIME<TAB>GIT_AUTHOR_DATE value
  local days="${DAILY_LOG_RANDOM_DAYS:-365}"
  python3 -c "
import random, time
from datetime import datetime, timezone
now = int(time.time())
start = now - int(${days}) * 86400
t = random.randint(start, now)
dt = datetime.fromtimestamp(t, tz=timezone.utc)
day = dt.strftime('%Y-%m-%d')
clock = dt.strftime('%H:%M')
git_d = dt.strftime('%Y-%m-%d %H:%M:%S') + ' +0000'
print(day + chr(9) + clock + chr(9) + git_d)
"
}

if [[ "${1:-}" == "--random-dated-commits" ]]; then
  shift
  N=""
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    N="$1"
    shift
  fi
  if [[ -z "$N" ]]; then
    N=$((2 + RANDOM % 5))
  fi
  if [[ "$N" -lt 1 ]]; then
    echo "N must be >= 1" >&2
    exit 1
  fi
  MSG="${*:-}"
  for ((i = 1; i <= N; i++)); do
    IFS=$'\t' read -r DAY TIME GIT_DATE < <(_random_line_and_git_date)
    if [[ -n "$MSG" ]]; then
      echo "- **$DAY** $TIME · \`$SHA\` $BR — $LINE — *$MSG*" >> "$LOG"
    else
      echo "- **$DAY** $TIME · \`$SHA\` $BR — $LINE" >> "$LOG"
    fi
    export GIT_AUTHOR_DATE="$GIT_DATE"
    export GIT_COMMITTER_DATE="$GIT_DATE"
    git add "$LOG"
    if git diff --staged --quiet; then
      echo "Nothing staged (unexpected)." >&2
      exit 1
    fi
    git commit -m "chore: daily log ($i/$N)"
  done
  echo "Created $N commit(s) on $LOG with random dates. Push when ready: git push"
  exit 0
fi

DAY=$(date +%Y-%m-%d)
TIME=$(date '+%H:%M')
MSG="${*:-}"
if [[ -n "$MSG" ]]; then
  echo "- **$DAY** $TIME · \`$SHA\` $BR — $LINE — *$MSG*" >> "$LOG"
else
  echo "- **$DAY** $TIME · \`$SHA\` $BR — $LINE" >> "$LOG"
fi
echo "Appended: $LOG"
