#!/bin/bash
# Generic cohort runner - runs the full 3-task suite for any model id.
# Usage: run_cohort.sh <model-id> [workers]        (OPENROUTER_API_KEY must be exported)
# Tasks: MILU n199 -> IndicQA n100 -> IndicXNLI n200.
# Skips tasks whose sheet already exists, so it is safe to re-invoke (resumable).
set -u
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
m="${1:?usage: run_cohort.sh <model-id> [workers]}"
w="${2:-8}"
slug="${m//\//_}"
echo "=== cohort: $m (workers=$w) ==="
if ! ls results/milu_${slug}_n19*.jsonl >/dev/null 2>&1; then
  python3 bench.py milu --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "^Accuracy|saved"
else echo "  milu: sheet exists, skip"; fi
if [ ! -f "results/indicqa_${slug}_n100.jsonl" ]; then
  python3 bench.py indicqa --model "$m" --n 100 --workers "$w" 2>/dev/null | grep -E "^EM|saved"
else echo "  indicqa: sheet exists, skip"; fi
if [ ! -f "results/xnli_${slug}_n200.jsonl" ]; then
  python3 bench.py xnli --model "$m" --n 200 --workers "$w" 2>/dev/null | grep -E "^Accuracy|Wrote"
else echo "  xnli: sheet exists, skip"; fi
python3 - "$slug" <<'PY'
import json, sys, glob
slug = sys.argv[1]
tot = 0.0
for f in sorted(glob.glob('results/*' + slug + '_n*.jsonl')):
    c = sum((json.loads(l).get('usage') or {}).get('cost') or 0.0 for l in open(f) if l.strip())
    if 'n5' in f:
        print('  %-52s $%.4f (smoke)' % (f.split('/')[-1], c))
    else:
        print('  %-52s $%.4f' % (f.split('/')[-1], c))
    tot += c
print('  TOTAL: $%.4f' % tot)
PY
