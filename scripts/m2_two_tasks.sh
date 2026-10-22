#!/bin/bash
# M2 stage 1 (P6_M2_TWO): ONE shared task-conditioned VLA on push_T + stack_bowls, <=2000 steps.
#
# Both tasks are REAL 100-episode datasets that are structurally identical to the frozen push_T
# contract (audited: schema diff empty over 300 episodes, A[t]==S[t+1] bit-exact). The per-task
# normalizers are required because the residual q99 spread across tasks is 2.44x; a single global
# normalizer would clip or quantise away most of stack_bowls' target range.
#
# save_every 200 because /home/T7 is ntfs-3g measured at 1.5-3 MB/s (one 1.1 GB save = 4-8 minutes);
# est-seconds-per-step 6.0 so the per-job wall guard (6.0 * 2000 = 12000 s) cannot cut the segment
# between save points at the measured ~3.1 s/step.
set -uo pipefail
cd /home/T7/ojh/robot_actual/r11
export PYTHONPATH=/home/dgut/pusht_runtime/openvla_oft_src:/home/dgut/pusht_runtime/openvla_oft_src/vla-scripts:${PYTHONPATH:-}
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/home/T7/ojh/robot_actual/runs/m2_two_2000_${STAMP}
LOG=logs/m2_two_2000_${STAMP}.log
NORM=data/m2_task_normalizer_2task.json
if [ ! -f "$NORM" ]; then echo "REFUSING: $NORM missing"; exit 9; fi
if [ -e "$OUT" ]; then echo "REFUSING: $OUT already exists"; exit 9; fi
echo "# M2 two-task $(date -u +%Y-%m-%dT%H:%M:%SZ) out=$OUT normalizer=$NORM"
"$PY" training/m2_train_shared.py --tasks push_T,stack_bowls --target-global-step 2000 \
  --out "$OUT" --save_every 200 --max-tasks-stage 2 \
  --est-seconds-per-step 6.0 --time-margin 300 --seed 0 --cams 3 \
  --bucket TWO_TASK_PILOT --splits data/m2_splits.json \
  --normalizer "$NORM" --dev-frames-per-task 8 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
ln -sfn "$OUT" /home/T7/ojh/robot_actual/runs/m2_two_latest
echo "# M2 two-task finished rc=$RC at $(date -u +%Y-%m-%dT%H:%M:%SZ); latest -> $OUT"
exit $RC
