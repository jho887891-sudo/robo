#!/bin/bash
# R15 stage 3: runs ONLY if the G5_1000 gate PASSED (the supervisor's dependency blocks it otherwise).
# Exactly +2000 optimizer steps, 1000 -> 3000, from the stage-2b step1000 checkpoint.
#
# --save_every is 200 because /home/T7 is ntfs-3g measured at 1.5-3 MB/s: one full-state save
# (~640 MB) costs 4-8 minutes, so 40 saves (save_every 50) would burn 5 hours of pure I/O.
# --est-seconds-per-step is deliberately pessimistic (8 s) so the per-job wall guard
# (est * need_steps = 16000 s) cannot cut the segment between save points; the real rate is ~3.4 s.
set -uo pipefail
cd /home/T7/ojh/robot_actual/r11
export PYTHONPATH=/home/dgut/pusht_runtime/openvla_oft_src:/home/dgut/pusht_runtime/openvla_oft_src/vla-scripts:${PYTHONPATH:-}
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
OUT=runs/r15_s3_3000_20261021T0758Z
LOG=logs/r15_s3_3000_20261021T0758Z.log
PARENT=runs/r15_s2b_1000_20261021T0758Z/checkpoints/step1000
if [ -e "$OUT" ]; then echo "REFUSING: $OUT already exists"; exit 9; fi
if [ ! -f "$PARENT/MANIFEST.json" ]; then echo "REFUSING: $PARENT missing"; exit 9; fi
echo "# stage3 $(date -u +%Y-%m-%dT%H:%M:%SZ) parent=$PARENT out=$OUT save_every=200 target=3000"
"$PY" training/r11_train_pushT_r42.py --r15 --target-global-step 3000 \
  --resume-from "$PARENT" --save_every 200 --out "$OUT" --bucket R15_TRAIN_FORMAL \
  --est-seconds-per-step 8.0 --time_margin 240 --lr 5e-4 --seed 0 --cams 3 \
  --run-id r15-s3-3000-20261021T0758Z 2>&1 | tee "$LOG"
exit ${PIPESTATUS[0]}
