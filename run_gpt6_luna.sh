#!/bin/bash
# Cohort runner - openai/gpt-6-luna (2026-10-XX): the GPT-6 successor to gpt-5.6-luna.
# Requires OPENROUTER_API_KEY already exported by the caller.
# Tasks per model: MILU n199 -> IndicQA n100 -> XNLI n200 (499 requests, ~$0.05).
# Skips existing sheets, so safe to re-invoke (resumable).
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
m="openai/gpt-6-luna"; w="${1:-8}"; slug="${m//\//_}"
if ! ls results/milu_${slug}_n19*.jsonl >/dev/null 2>&1; then
  echo "[$(date +%F' '%T)] MILU $m (workers=$w)"
  python3 bench.py milu --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "Accuracy|MILU-Tamil|saved"
fi
if [ ! -f "results/indicqa_${slug}_n100.jsonl" ]; then
  echo "[$(date +%F' '%T)] IndicQA $m"
  python3 bench.py indicqa --model "$m" --n 100 --workers "$w" 2>/dev/null | grep -E "^EM|saved"
fi
if ! ls results/xnli_${slug}_n200.jsonl >/dev/null 2>&1; then
  echo "[$(date +%F' '%T)] XNLI $m"
  python3 bench.py xnli --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "Accuracy|Wrote"
fi
echo "[$(date +%F' '%T)] COHORT_DONE $m"
