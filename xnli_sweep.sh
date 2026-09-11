#!/bin/bash
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
MODELS=(
  "google/gemini-3.8-flash" "openai/gpt-5.6-luna" "deepseek/deepseek-v4.1-flash"
  "z-ai/glm-5.3-flash" "qwen/qwen3.8-flash" "xiaomi/mimo-v2.5"
  "google/gemma-4-26b-a4b-it:free" "inclusionai/ling-3.0-flash-vl:free"
  "nvidia/nemotron-3.5-lightning:free"
)
for m in "${MODELS[@]}"; do
  echo "[$(date +%H:%M:%S)] XNLI $m" >> xnli_sweep.log
  python3 bench.py xnli --model "$m" --n 200 --workers 8 >> xnli_sweep.log 2>&1
done
echo "[$(date +%H:%M:%S)] SWEEP DONE" >> xnli_sweep.log
