#!/bin/bash
# R15 stage 2b: continue from the LAST checkpoint the first segment managed to save, up to 1000.
#
# Why this exists: on this host /home/T7 is an ntfs-3g volume measured at 1.5-3 MB/s, so one
# full-state save (~640 MB) takes 4-8 minutes. The per-job wall guard
# (est_seconds_per_step * need_steps) therefore ends the segment between save points, and a few
# optimizer steps are spent and discarded. This segment resumes the exact saved state, so the
# resulting step1000 checkpoint is exactly +700 optimizer steps after the verified step300 parent.
# --save_every is raised to 200 because checkpoint I/O, not compute, is the wall-clock bottleneck.
set -uo pipefail
cd /home/T7/ojh/robot_actual/r11
export PYTHONPATH=/home/dgut/pusht_runtime/openvla_oft_src:/home/dgut/pusht_runtime/openvla_oft_src/vla-scripts:${PYTHONPATH:-}
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
SRC=runs/r15_s2_1000_20261021T0803Z/checkpoints
OUT=runs/r15_s2b_1000_20261021T0803Z
LOG=logs/r15_s2b_1000_20261021T0803Z.log
if [ ! -d "$SRC" ]; then echo "REFUSING: $SRC missing"; exit 9; fi
LAST=$(ls -1d "$SRC"/step* 2>/dev/null | sed 's#.*/step##' | sort -n | tail -1)
if [ -z "${LAST:-}" ]; then echo "REFUSING: no checkpoint saved in $SRC"; exit 9; fi
if [ "$LAST" -le 300 ]; then echo "REFUSING: last checkpoint step $LAST did not advance past the parent"; exit 9; fi
if [ -e "$OUT" ]; then echo "REFUSING: $OUT already exists"; exit 9; fi
echo "# stage2b $(date -u +%Y-%m-%dT%H:%M:%SZ) parent=$SRC/step$LAST target=1000 out=$OUT save_every=200"
"$PY" training/r11_train_pushT_r42.py --r15 --target-global-step 1000 \
  --resume-from "$SRC/step$LAST" --save_every 200 --out "$OUT" --bucket R15_TRAIN_FORMAL \
  --est-seconds-per-step 12.0 --time_margin 120 --lr 5e-4 --seed 0 --cams 3 \
  --run-id r15-s2b-1000-20261021T0803Z 2>&1 | tee "$LOG"
exit ${PIPESTATUS[0]}
