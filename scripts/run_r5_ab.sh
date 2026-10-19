#!/usr/bin/env bash
# R5: read-only preflight by default. GPU training requires explicit --execute.
set -Eeuo pipefail
if [[ $# -gt 1 ]]; then echo "Usage: bash scripts/run_r5_ab.sh [--check|--execute]" >&2; exit 2; fi
mode="${1:---check}"
if [[ "$mode" != "--check" && "$mode" != "--execute" ]]; then
  echo "Usage: bash scripts/run_r5_ab.sh [--check|--execute]" >&2; exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if [[ "$(git branch --show-current)" != "audit/r5-distribution-and-baseline-audit" ]]; then
  echo "[STOP] Use a worktree on audit/r5-distribution-and-baseline-audit" >&2; exit 3
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[STOP] Dirty tracked working tree; use a clean worktree" >&2; exit 3
fi
echo "[preflight] commit=$(git rev-parse HEAD)"
export PUSHT_RUNTIME="${PUSHT_RUNTIME:-/home/dgut/pusht_runtime}"
export PUSHT_MODEL="${PUSHT_MODEL:-$PUSHT_RUNTIME/model_cache/openvla-7b-pristine}"
export PUSHT_HDF5="${PUSHT_HDF5:-/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data}"
export PUSHT_OFT="${PUSHT_OFT:-/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft}"
export PUSHT_SPLIT="${PUSHT_SPLIT:-$PUSHT_RUNTIME/split80.json}"
export PUSHT_SAMPLES="${PUSHT_SAMPLES:-$PUSHT_RUNTIME/fixed_samples.json}"
export PUSHT_BOUNDS="${PUSHT_BOUNDS:-$PUSHT_RUNTIME/push_T_bounds.npz}"
py="${R5_PYTHON:-$PUSHT_RUNTIME/venv_oft/bin/python}"
if [[ ! -x "$py" ]]; then echo "[STOP] Python not executable: $py" >&2; exit 4; fi
"$py" - "$root" <<'PY'
import hashlib, json, os, pathlib, sys
repo = pathlib.Path(sys.argv[1])
model = pathlib.Path(os.environ["PUSHT_MODEL"])
for key in ("PUSHT_SPLIT", "PUSHT_SAMPLES", "PUSHT_BOUNDS"):
    p = pathlib.Path(os.environ[key])
    if not p.is_file():
        raise SystemExit(f"[STOP] Missing {key}: {p}")
for key in ("PUSHT_HDF5", "PUSHT_OFT", "PUSHT_MODEL"):
    p = pathlib.Path(os.environ[key])
    if not p.is_dir():
        raise SystemExit(f"[STOP] Missing {key}: {p}")
a = repo / "code/openvla_dynamic_modules/modeling_prismatic.py"
b = model / "modeling_prismatic.py"
if not (a.is_file() and b.is_file()):
    raise SystemExit("[STOP] Missing pristine dynamic modeling module")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(a) != sha(b):
    raise SystemExit("[STOP] Active base source differs from pristine verified R3 source")
print("[preflight] pristine modeling sha256:", sha(a))
split = json.loads(pathlib.Path(os.environ["PUSHT_SPLIT"]).read_text())
train, val = set(split["train"]), set(split["val"])
if not train or not val or train & val:
    raise SystemExit("[STOP] Train/val episodes empty or overlapping")
fixed = json.loads(pathlib.Path(os.environ["PUSHT_SAMPLES"]).read_text())
if len(fixed) < 20 or any(e not in val for e, t in fixed[:20]):
    raise SystemExit("[STOP] Fixed held-out samples are not in validation split")
for e in train | val:
    if not (pathlib.Path(os.environ["PUSHT_HDF5"]) / e).is_file():
        raise SystemExit("[STOP] Missing episode data: " + e)
print(f"[preflight] {len(train)} TRAIN episodes / {len(val)} VAL episodes; 60 spread held-out frames at training time")
PY
"$py" -m py_compile training/train_pushT_pure2.py training/overfit_pr1.py \
    training/compare_target_runs.py code/xpolicylab/policy/OpenVLA_Pure/model.py
"$py" tests/test_action_target_contract.py
"$py" tests/test_target_ab_compare.py
"$py" tests/test_overfit_protocol.py
"$py" tests/test_action_codec.py
"$py" tests/test_r5_distribution_metrics.py
"$py" diagnostics/audit_r4_evidence.py
echo "[preflight] CPU tests passed"
if [[ "$mode" == "--check" ]]; then
  echo "[DONE] Preflight only: GPU training NOT started."
  exit 0
fi
"$py" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[STOP] CUDA not available")
free, total = torch.cuda.mem_get_info()
print(f"[preflight] GPU free {free/1e9:.1f}GB of {total/1e9:.1f}GB")
if free < 20e9:
    raise SystemExit("[STOP] Less than 20GB free VRAM")
PY
run_dir="${R5_RUN_DIR:-$PUSHT_RUNTIME/r5_ab_jobs/$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$run_dir" ]]; then echo "[STOP] Output directory already exists: $run_dir" >&2; exit 5; fi
mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/commit_sha.txt"
echo "[run] absolute target (train 200 from 80 episodes, validation 60 across 20 episodes, 800 updates)"
"$py" training/overfit_pr1.py --mechanism keep --target-mode absolute --n 200 --val-n 60 --steps 800 --sample-strategy diverse --val-strategy spread \
    --eval-every 200 --out "$run_dir/absolute_keep" 2>&1 | tee "$run_dir/absolute_console.log"
echo "[run] residual target (same split, seed, setup)"
"$py" training/overfit_pr1.py --mechanism keep --target-mode residual --n 200 --val-n 60 --steps 800 --sample-strategy diverse --val-strategy spread \
    --eval-every 200 --out "$run_dir/residual_keep" 2>&1 | tee "$run_dir/residual_console.log"
"$py" training/compare_target_runs.py --absolute "$run_dir/absolute_keep" \
    --residual "$run_dir/residual_keep" --out "$run_dir/target_comparison.json" \
    2>&1 | tee "$run_dir/compare_console.log"
echo "[DONE] $run_dir/target_comparison.json"
