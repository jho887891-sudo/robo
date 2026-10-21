#!/bin/bash
# G5 DEV predictions at step1000 on the frozen 32-frame gate list (no gradient, no update).
# Reads the step1000 checkpoint produced by stage 2b (NOT stage 2a, which stopped at step500).
# The 7B inference wall clock is booked into the WAL: the audit found this GPU time unbilled before.
set -uo pipefail
cd /home/T7/ojh/robot_actual/r11
export PYTHONPATH=/home/dgut/pusht_runtime/openvla_oft_src:/home/dgut/pusht_runtime/openvla_oft_src/vla-scripts:${PYTHONPATH:-}
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
CK=runs/r15_s2b_1000_20261021T0803Z/checkpoints/step1000
NPZ=runs/r15_dev_gate_1000_20261021T0803Z.json
LOG=logs/r15_devpreds_1000_20261021T0803Z.log
if [ ! -f "$CK/MANIFEST.json" ]; then echo "REFUSING: $CK missing (stage 2b did not reach step1000)"; exit 9; fi
S=$(date +%s)
"$PY" training/r11_train_pushT_r42.py --dev-preds "$NPZ" --resume-from "$CK" \
  --dev-per-episode 2 --dev-seed 0 --out runs/r15_devpreds_1000_run_20261021T0803Z --cams 3 --seed 0 \
  2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
E=$(( $(date +%s) - S ))
"$PY" - "$RC" "$E" <<'PYEOF'
import sys
sys.path.insert(0, "training")
import r15_ledger as L
rc, secs = int(sys.argv[1]), float(sys.argv[2])
with L.Ledger() as led:
    led.append("r15-devpred-1000-20261021T0803Z", "job_end", bucket="R15_TRAIN_FORMAL",
               payload={"what": "DEV 7B greedy 32-frame inference for gate G5_1000",
                        "exit": rc, "counts_toward_total": True},
               gpu_seconds=secs, formal_steps=0)
print("booked DEV inference %.0fs (exit %d)" % (secs, rc))
PYEOF
exit $RC
