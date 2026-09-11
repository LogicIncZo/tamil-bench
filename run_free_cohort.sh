#!/bin/bash
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
run() {
  m="$1"; w="$2"; slug="${m//\//_}"
  if ! ls results/milu_${slug}_n19*.jsonl >/dev/null 2>&1 && ! ls results/milu_${slug}_n200.jsonl >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] MILU $m (workers=$w)"
    python3 bench.py milu --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "^Accuracy"
  fi
  if [ ! -f "results/indicqa_${slug}_n100.jsonl" ]; then
    echo "[$(date +%H:%M:%S)] IndicQA $m"
    python3 bench.py indicqa --model "$m" --n 100 --workers "$w" 2>/dev/null | grep -E "^EM"
  fi
}
run "google/gemma-4-26b-a4b-it:free" 8
run "inclusionai/ling-3.0-flash-vl:free" 8
run "poolside/laguna-s-2.1:free" 4
run "nvidia/nemotron-3.5-lightning:free" 4
echo "ALL_DONE"
