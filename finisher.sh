#!/bin/bash
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
LOG=finisher.log
echo "[$(date +%H:%M:%S)] waiting for xnli sweep" >> $LOG
while ! grep -q "SWEEP DONE" xnli_sweep.log; do sleep 60; done
echo "[$(date +%H:%M:%S)] sweep done; running nemotron milu n199" >> $LOG
for i in 1 2 3 4 5; do
  python3 bench.py milu --model nvidia/nemotron-3.5-lightning:free --n 199 --workers 6 >> $LOG 2>&1 && break
  echo "[$(date +%H:%M:%S)] milu attempt $i failed; retry in 180s" >> $LOG; sleep 180
done
echo "[$(date +%H:%M:%S)] running nemotron indicqa n100" >> $LOG
for i in 1 2 3 4 5; do
  python3 bench.py indicqa --model nvidia/nemotron-3.5-lightning:free --n 100 --workers 6 >> $LOG 2>&1 && break
  echo "[$(date +%H:%M:%S)] indicqa attempt $i failed; retry in 180s" >> $LOG; sleep 180
done
echo "[$(date +%H:%M:%S)] build + push" >> $LOG
python3 build_site.py >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] FINISHER DONE" >> $LOG
