#!/usr/bin/env bash
# Quality gate.
#
# The gate itself lives outside this repo. Point $QA_GATE at it, or put the
# path in .qa-gate-path beside this script (untracked).
#
#   ./check.sh                    full gate
#   ./check.sh --fast             static analysis only, skip tests
#   ./check.sh --changed          only findings in files you have touched
#   ./check.sh --update-baseline  accept current findings; commit the result
set -euo pipefail
cd "$(dirname "$0")"
QA_GATE="${QA_GATE:-$(cat .qa-gate-path 2>/dev/null)}"
if [ -z "$QA_GATE" ]; then
  echo "check.sh: set \$QA_GATE or write the qa-gate path to .qa-gate-path" >&2
  exit 2
fi
exec python3 "$QA_GATE/qa_check.py" "$@"
