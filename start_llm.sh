~/AI/llama.cpp/build/bin/llama-server \
  -m ~/AI/llama.cpp/models/Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q4_K_M.gguf \
  --alias qwen35-9b \
  -c 24000 \
  -t 4 --temp 1.2 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --load-mode auto \
  --host 0.0.0.0 \
  --port 8080 &

sleep 15
python -m bot
