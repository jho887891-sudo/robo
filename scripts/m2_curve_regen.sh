#!/usr/bin/env bash
# Regenerate the M2 DEV CE curve at every save point with the CORRECTED action-window alignment.
# The curve stored in stage_evidence.json was measured with the vision-prefix bug, so its absolute values
# are not comparable to the corrected final point (erratum E5). This walks the sealed checkpoints, never
# retrains, and writes one archive + report per save point plus an aggregated corrected curve.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
RUN=/home/T7/ojh/robot_actual/runs/m2_two_2000_20261021T123314Z
OUTD="${RUN}/curve_v2"
mkdir -p "$OUTD"
mkdir -p mission
for CK in "$RUN"/checkpoints/step*; do
  [ -d "$CK" ] || continue
  STEP=$(basename "$CK" | sed 's/step//')
  echo "=== step $STEP ==="
  $PY tools/m2_regen_dev.py --checkpoint "$CK" --tasks push_T,stack_bowls \
      --normalizer data/m2_task_normalizer_2task.json --splits data/m2_splits.json \
      --dev-frames-per-task 8 --cams 3 --seed 0 \
      --out "$OUTD/dev_eval_step${STEP}.npz" \
      --report "$OUTD/regen_step${STEP}.json" || echo "STEP $STEP FAILED"
done
$PY - <<'PYEOF'
import glob, json, os
rows = []
for p in sorted(glob.glob("/home/T7/ojh/robot_actual/runs/m2_two_2000_20261021T123314Z/curve_v2/regen_step*.json")):
    d = json.load(open(p))
    rows.append({"global_step": d["checkpoint_global_step"],
                 "ce_window": d["ce_window"], "ce_full": d["ce_full"],
                 "archive_sha256": d["archive_sha256"],
                 "alignment": "vision-prefix-aware-v2"})
rows.sort(key=lambda r: r["global_step"])
out = {"schema": "m2-ce-curve-corrected-v1",
       "why": ("the stage_evidence curve used the legacy action-window alignment; this is the same "
               "curve recomputed on the sealed checkpoints with the corrected one (erratum E5)"),
       "no_retraining": True, "points": rows}
p = "/home/T7/ojh/robot_actual/r11/mission/m2_ce_curve_corrected.json"
json.dump(out, open(p, "w"), indent=1)
for r in rows:
    print(r["global_step"], {k: round(v, 4) for k, v in r["ce_window"].items()})
print("wrote", p)
PYEOF
