#!/bin/bash
# Full suite for x-ai/grok-4.7: MILU n199 + IndicQA n100 + XNLI n200 (seed 42).
# Resumable — each sheet is skipped if it already exists.
cd "$(dirname "$0")" || exit 1

MODEL="x-ai/grok-4.7"
SAFE="$(echo "$MODEL" | tr '/:' '__')"

run () {
  task="$1"; n="$2"
  if [ -f "results/${task}_${SAFE}_n${n}.jsonl" ]; then
    echo "=== skip ${task} n=${n} (sheet exists) ==="
    return
  fi
  echo "=== ${MODEL} ${task} n=${n} ==="
  python3 bench.py "$task" --model "$MODEL" --n "$n"
}

run milu 199
run indicqa 100
run xnli 200
echo "=== done ==="