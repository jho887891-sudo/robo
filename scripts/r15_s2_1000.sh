#!/bin/bash
# R15 stage 2: exact continuation step300 -> 1000 (+700 optimizer steps).
# Parent = the verified step300 checkpoint of the fresh 0->300 run (not polluted by the resume bug).
# New run directory only; existing evidence is never touched.
# NOTE: the platform token (arx_x5_push_T) is NOT passed here - argparse rejects it; the trainer's
# force_arx_platform() appends it to sys.argv itself before any prismatic import (fail-closed).
set -uo pipefail
cd /home/T7/ojh/robot_actual/r11
export PYTHONPATH=/home/dgut/pusht_runtime/openvla_oft_src:/home/dgut/pusht_runtime/openvla_oft_src/vla-scripts:${PYTHONPATH:-}
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
OUT=runs/r15_s2_1000_20261021T0803Z
LOG=logs/r15_s2_1000_20261021T0803Z.log
PARENT=runs/r15_stage_300_20261021T030423Z/checkpoints/step300
if [ -e "$OUT" ]; then echo "REFUSING: $OUT already exists"; exit 9; fi
if [ ! -f "$PARENT/MANIFEST.json" ]; then echo "REFUSING: parent checkpoint missing"; exit 9; fi
echo "# stage2 $(date -u +%Y-%m-%dT%H:%M:%SZ) parent=$PARENT out=$OUT"
"$PY" training/r11_train_pushT_r42.py --r15 --target-global-step 1000 \
  --resume-from "$PARENT" --save_every 50 --out "$OUT" --bucket R15_TRAIN_FORMAL \
  --est-seconds-per-step 4.0 --time_margin 180 --lr 5e-4 --seed 0 --cams 3 \
  --run-id r15-s2-1000-20261021T0803Z 2>&1 | tee "$LOG"
exit ${PIPESTATUS[0]}
