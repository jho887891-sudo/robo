#!/bin/bash
# Adopt an already-running job into the supervisor DAG: block until its artifact exists.
# Used because the step300->1000 trainer was started while the supervisor itself was being fixed;
# it keeps the GPU lane lock, so nothing else can run until it exits either way.
set -uo pipefail
F=${1:?artifact path}
LIMIT=${2:-4200}
S=$(date +%s)
while [ ! -f "$F" ]; do
  NOW=$(date +%s)
  if [ $((NOW - S)) -gt "$LIMIT" ]; then
    echo "TIMEOUT waiting for $F after ${LIMIT}s"
    exit 7
  fi
  if ! ps -eo args | grep -q '[r]11_train_pushT_r42.py'; then
    sleep 10
    if [ ! -f "$F" ]; then
      echo "NO TRAINER ALIVE and $F is missing -> the stage did not produce its checkpoint"
      exit 8
    fi
  fi
  sleep 10
done
echo "artifact present: $F ($(( $(date +%s) - S ))s)"
ls -l "$F"
