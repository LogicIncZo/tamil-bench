#!/bin/bash
# Sub-dollar free cohort #2 (2026-09-29): 5 new :free models.
# Resumable: skips any model that already has result sheets.
# 499 requests per full suite (MILU 199 + IndicQA 100 + XNLI 200);
# OpenRouter free tier = 1000 req/day key-wide, so ~2 suites complete per day.
# Re-run daily until ALL_DONE printed with nothing left.
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
run() {
  m="$1"; w="${2:-4}"; slug="${m//\//_}"
  if ! ls results/milu_${slug}_n19*.jsonl >/dev/null 2>&1; then
    echo "[$(date +%F' '%T)] MILU $m (workers=$w)"
    python3 bench.py milu --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "Accuracy|MILU-Tamil"
  fi
  if [ ! -f "results/indicqa_${slug}_n100.jsonl" ]; then
    echo "[$(date +%F' '%T)] IndicQA $m"
    python3 bench.py indicqa --model "$m" --n 100 --workers "$w" 2>/dev/null | grep -E "EM:"
  fi
  if ! ls results/xnli_${slug}_n*.jsonl >/dev/null 2>&1; then
    echo "[$(date +%F' '%T)] XNLI $m"
    python3 bench.py xnli --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "Accuracy|acc"
  fi
  echo "[$(date +%F' '%T)] done: $m"
}
run "nvidia/nemotron-3-ultra-550b-a55b:free" 4
run "nvidia/nemotron-3-super-120b-a12b:free" 4
run "inclusionai/ling-3.0-flash-sante:free" 4
run "liquid/lfm-2.5-2.6b:free" 4
run "poolside/laguna-xs-2.1:free" 4
echo "ALL_DONE free_cohort2"
