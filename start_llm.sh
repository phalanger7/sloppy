#!/usr/bin/env bash
# Start llama.cpp with the bot's model, then run the bot against it.
# Stopping this script (Ctrl-C, or any exit) also stops the server.
set -euo pipefail

MODEL="$HOME/AI/models/Occult Nail/Occult-Nail-1.0-35B-A3B-UD-Q4_K_XL.gguf"
PORT=8080

~/AI/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --alias OccultNail \
  -c 32000 -np 4 \
  -t 4 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --load-mode auto \
  --host 0.0.0.0 \
  --port "$PORT" &
LLAMA_PID=$!
trap 'kill "$LLAMA_PID" 2>/dev/null || true' EXIT

# Wait for the model to finish loading rather than guessing at a fixed sleep --
# a cold page cache can take well over a minute.
echo "waiting for llama-server on :$PORT ..."
for _ in $(seq 1 120); do
  if curl -sf -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "server ready"
    break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "llama-server exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done

python3 llmbot_tui.py
