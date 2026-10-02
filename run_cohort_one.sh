#!/bin/bash
# Cohort runner — one model per invocation: bash run_cohort_one.sh <model> [workers]
# Tasks per model: MILU n200 -> IndicQA n100 -> XNLI n200 (499 requests/suite).
# Skips existing sheets, so safe to re-invoke (resumable across free-tier daily caps).
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
m="$1"; w="${2:-4}"; slug="${m//\//_}"
if ! ls results/milu_${slug}_n19*.jsonl >/dev/null 2>&1 && ! ls results/milu_${slug}_n200.jsonl >/dev/null 2>&1; then
  echo "[$(date +%H:%M:%S)] MILU $m (workers=$w)"
  python3 bench.py milu --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "^Accuracy"
fi
if [ ! -f "results/indicqa_${slug}_n100.jsonl" ]; then
  echo "[$(date +%H:%M:%S)] IndicQA $m"
  python3 bench.py indicqa --model "$m" --n 100 --workers "$w" 2>/dev/null | grep -E "^EM"
fi
if [ ! -f "results/xnli_${slug}_n200.jsonl" ]; then
  echo "[$(date +%H:%M:%S)] XNLI $m"
  python3 bench.py xnli --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "^Accuracy"
fi
echo "[$(date +%H:%M:%S)] COHORT_DONE $m"
