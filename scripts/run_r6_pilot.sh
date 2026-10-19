#!/usr/bin/env bash
# R6 history-proprio pilot runner.
#   bash scripts/run_r6_pilot.sh --check     read-only preflight + Phase-1 CPU diagnostics
#   bash scripts/run_r6_pilot.sh --smoke     GPU smoke test gate (28D interface)
#   bash scripts/run_r6_pilot.sh --execute   300-update A/B on GPU (requires smoke PASS)
# Nothing runs on the GPU unless the explicit flag is given. No closed loop, no
# 3000/6000-step training, no R2-R5 checkpoint or dataset is touched.
set -Eeuo pipefail
mode="${1:---check}"
case "$mode" in
  --check|--smoke|--execute) ;;
  *) echo "Usage: bash scripts/run_r6_pilot.sh [--check|--smoke|--execute]" >&2; exit 2 ;;
esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if [[ "$(git branch --show-current)" != "audit/r6-history-proprio-pilot" ]]; then
  echo "[STOP] Use a worktree on audit/r6-history-proprio-pilot" >&2; exit 3
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[STOP] Dirty tracked working tree; commit before running" >&2; exit 3
fi
echo "[preflight] commit=$(git rev-parse HEAD) mode=$mode"
export PUSHT_RUNTIME="${PUSHT_RUNTIME:-/home/dgut/pusht_runtime}"
export PUSHT_MODEL="${PUSHT_MODEL:-$PUSHT_RUNTIME/model_cache/openvla-7b-pristine}"
export PUSHT_HDF5="${PUSHT_HDF5:-/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data}"
export PUSHT_OFT="${PUSHT_OFT:-/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft}"
export PUSHT_SPLIT="${PUSHT_SPLIT:-$PUSHT_RUNTIME/split80.json}"
export PUSHT_BOUNDS="${PUSHT_BOUNDS:-$PUSHT_RUNTIME/push_T_bounds.npz}"
export R6_TS="${R6_TS:-20261019T161033Z}"
export R6_EVIDENCE="${R6_EVIDENCE:-$root/evidence/history_proprio_pilot_$R6_TS}"
export R6_RUNS="${R6_RUNS:-$PUSHT_RUNTIME/r6_pilot_runs/$R6_TS}"
py="${R6_PYTHON:-$PUSHT_RUNTIME/venv_oft/bin/python}"
[[ -x "$py" ]] || { echo "[STOP] Python not executable: $py" >&2; exit 4; }
mkdir -p "$R6_EVIDENCE" "$R6_RUNS"

"$py" - <<'PY'
import hashlib, json, os, pathlib
repo = pathlib.Path(os.environ.get("PWD", "."))
model = pathlib.Path(os.environ["PUSHT_MODEL"])
for key in ("PUSHT_SPLIT", "PUSHT_BOUNDS"):
    p = pathlib.Path(os.environ[key])
    if not p.is_file():
        raise SystemExit("[STOP] Missing %s: %s" % (key, p))
for key in ("PUSHT_HDF5", "PUSHT_OFT", "PUSHT_MODEL"):
    if not pathlib.Path(os.environ[key]).is_dir():
        raise SystemExit("[STOP] Missing %s" % key)
a = repo / "code/openvla_dynamic_modules/modeling_prismatic.py"
b = model / "modeling_prismatic.py"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if not (a.is_file() and b.is_file()) or sha(a) != sha(b):
    raise SystemExit("[STOP] Active base modeling source differs from the pristine verified module")
split = json.loads(pathlib.Path(os.environ["PUSHT_SPLIT"]).read_text())
train, val = set(split["train"]), set(split["val"])
if not train or not val or train & val:
    raise SystemExit("[STOP] train/val split empty or overlapping")
for e in train | val:
    if not (pathlib.Path(os.environ["PUSHT_HDF5"]) / e).is_file():
        raise SystemExit("[STOP] Missing episode data: " + e)
print("[preflight] pristine modeling sha256:", sha(a))
print("[preflight] %d TRAIN / %d VAL episodes" % (len(train), len(val)))
PY

"$py" -m py_compile training/r6_history_pilot.py training/r6_metrics.py training/compare_history_arms.py \
  diagnostics/r6_data.py diagnostics/r6_align.py diagnostics/r6_history_value.py \
  diagnostics/r6_quant_oracle.py diagnostics/r6_smoke.py \
  code/xpolicylab/policy/OpenVLA_Pure/proprio_history.py
echo "[preflight] py_compile OK"
"$py" tests/test_r6_proprio_history.py 2>&1 | tail -3
"$py" tests/test_r6_history_diagnostics.py 2>&1 | tail -3

echo "[phase1] real-HDF5 diagnostics -> $R6_EVIDENCE"
"$py" diagnostics/r6_align.py --out "$R6_EVIDENCE/phase1_alignment.json" | tail -12
"$py" diagnostics/r6_history_value.py --out "$R6_EVIDENCE/phase1_history_value.json" | tail -14
"$py" diagnostics/r6_quant_oracle.py --out "$R6_EVIDENCE/phase1_quant_oracle.json" | tail -14
git rev-parse HEAD > "$R6_EVIDENCE/commit_sha.txt"

if [[ "$mode" == "--check" ]]; then
  echo "[check] preflight + Phase-1 diagnostics complete; nothing was launched on the GPU"
  exit 0
fi

if [[ "$mode" == "--smoke" ]]; then
  "$py" diagnostics/r6_smoke.py --history-mode true --out "$R6_EVIDENCE/smoke_true.json" 2>&1 | tail -30
  "$py" diagnostics/r6_smoke.py --history-mode duplicate --out "$R6_EVIDENCE/smoke_duplicate.json" 2>&1 | tail -30
  echo "[smoke] artifacts in $R6_EVIDENCE"
  exit 0
fi

if [[ ! -f "$R6_EVIDENCE/smoke_true.json" ]]; then
  echo "[STOP] --execute requires a PASSing smoke gate first: bash scripts/run_r6_pilot.sh --smoke" >&2; exit 6
fi
"$py" - "$R6_EVIDENCE/smoke_true.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("result") != "PASS":
    raise SystemExit("[STOP] smoke gate did not pass; refusing to train")
print("[preflight] smoke gate PASS (%d checks)" % len(d["checks"]))
PY

for arm in duplicate true; do
  out="$R6_RUNS/${arm}_residual"
  if [[ -e "$out" ]]; then echo "[STOP] run directory already exists: $out" >&2; exit 7; fi
  echo "[execute] arm=$arm -> $out"
  "$py" training/r6_history_pilot.py --history-mode "$arm" --target-mode residual \
    --n 200 --val-n 60 --steps 300 --lr 5e-4 --eval-every 150 \
    --out "$out" 2>&1 | tee "$R6_EVIDENCE/train_${arm}.log" | tail -40
done

"$py" training/compare_history_arms.py \
  --duplicate "$R6_RUNS/duplicate_residual" --true "$R6_RUNS/true_residual" \
  --phase1-dir "$R6_EVIDENCE" --out "$R6_EVIDENCE/arm_comparison.json" | tee "$R6_EVIDENCE/compare.log"
echo "[execute] done; evidence in $R6_EVIDENCE"
