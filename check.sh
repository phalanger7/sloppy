#!/usr/bin/env bash
# Quality gate. See /home/phloid/AI/tooling/qa-gate/README.md
#
#   ./check.sh                    full gate
#   ./check.sh --fast             static analysis only, skip tests
#   ./check.sh --changed          only findings in files you have touched
#   ./check.sh --update-baseline  accept current findings; commit the result
set -euo pipefail
cd "$(dirname "$0")"
exec python3 "${QA_GATE:-/home/phloid/AI/tooling/qa-gate}/qa_check.py" "$@"
