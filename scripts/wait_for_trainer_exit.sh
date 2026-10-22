#!/bin/bash
# Wait until no R15 trainer process is alive, then require that at least one checkpoint was saved.
# This is how an already-running job is adopted into the supervisor DAG: the supervisor waits for the
# job's ARTIFACT, it never starts a second trainer on the same GPU.
set -uo pipefail
DIR=${1:?checkpoint dir}
LIMIT=${2:-4200}
S=$(date +%s)
while ps -eo args | grep -q '[r]11_train_pushT_r42.py'; do
  NOW=$(date +%s)
  if [ $((NOW - S)) -gt "$LIMIT" ]; then echo "TIMEOUT: trainer still alive after ${LIMIT}s"; exit 7; fi
  sleep 10
done
N=$(ls -1d "$DIR"/step* 2>/dev/null | wc -l)
if [ "$N" -lt 1 ]; then echo "NO CHECKPOINT SAVED in $DIR"; exit 8; fi
echo "trainer exited; $N checkpoints present in $DIR:"
ls -1d "$DIR"/step* | sed 's#.*/##' | sort -V | tr '\n' ' '
echo ""
