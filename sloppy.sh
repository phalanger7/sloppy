#!/usr/bin/env bash
# Start the TUI under tmux, or attach to it if it is already running.
#
#   ./sloppy.sh            start it detached, then attach
#   ./sloppy.sh --detach   start it and leave it in the background
#   ./sloppy.sh --stop     stop the session
#   ./sloppy.sh --status   is it running
#
# ctrl-b d leaves it running and gives you the terminal back.
#
# This is the "I want a UI I can come back to" option. For something that
# survives a reboot and restarts itself, use the systemd unit instead -- see
# sloppy.service.example.
set -euo pipefail
cd "$(dirname "$0")"

SESSION="${SLOPPY_TMUX_SESSION:-sloppy}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "sloppy.sh: tmux is not installed" >&2
  exit 1
fi

running() { tmux has-session -t "=$SESSION" 2>/dev/null; }

# Attaching from inside tmux would nest a session inside itself, which tmux
# refuses; switch the current client to it instead.
attach() {
  if [ -n "${TMUX:-}" ]; then
    tmux switch-client -t "=$SESSION"
  else
    tmux attach-session -t "=$SESSION"
  fi
}

case "${1:-}" in
  --stop)
    running || { echo "not running"; exit 0; }
    tmux kill-session -t "=$SESSION"
    echo "stopped"
    exit 0
    ;;
  --status)
    running && echo "running (tmux session '$SESSION')" || echo "not running"
    exit 0
    ;;
  --detach|--attach|"") ;;
  *)
    echo "sloppy.sh: unknown option '$1'" >&2
    exit 2
    ;;
esac

if running; then
  # Already up: never start a second bot on the same channel, just go to it.
  [ "${1:-}" = "--detach" ] && { echo "already running"; exit 0; }
  attach
  exit 0
fi

tmux new-session -d -s "$SESSION" "python3 llmbot_tui.py"
if [ "${1:-}" = "--detach" ]; then
  echo "started in tmux session '$SESSION'; ./sloppy.sh to attach"
  exit 0
fi
attach
