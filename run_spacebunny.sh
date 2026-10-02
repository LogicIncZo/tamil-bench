#!/bin/bash
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
for t in "milu 199" "indicqa 100" "xnli 200"; do
  set -- $t
  echo "=== space-bunny-alpha $1 n=$2 ==="
  python3 bench.py "$1" --model stealth/space-bunny-alpha --n "$2" 2>&1
done
echo "DONE space-bunny suite"
