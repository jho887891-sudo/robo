#!/usr/bin/env bash
# R7 state-dynamics baseline runner (CPU only; no OpenVLA, no GPU, no closed loop).
#   bash scripts/run_r7.sh --check   unit tests + phase-1 oracle diagnostics
#   bash scripts/run_r7.sh --mlp     train the MLP grid + acceleration experiment
#   bash scripts/run_r7.sh --probe   phase-3 diagnostics + independent verification
#   bash scripts/run_r7.sh --all     everything, in order
set -Eeuo pipefail
mode="${1:---check}"
case "$mode" in
  --check|--mlp|--probe|--all) ;;
  *) echo "Usage: bash scripts/run_r7.sh [--check|--mlp|--probe|--all]" >&2; exit 2 ;;
esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
current_branch="$(git branch --show-current)"
case "$current_branch" in
  audit/r7-state-dynamics-baseline|review/r7-state-dynamics-audit-20261020) ;;
  *) echo "[STOP] use an isolated R7 baseline/audit worktree" >&2; exit 3 ;;
esac
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[STOP] dirty tracked tree; commit first" >&2; exit 3
fi
export PUSHT_HDF5="${PUSHT_HDF5:-/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data}"
export PUSHT_SPLIT="${PUSHT_SPLIT:-/home/dgut/pusht_runtime/split80.json}"
export R7_TS="${R7_TS:-20261020T0050Z_review}"
# Do not overwrite the R7 baseline run with corrected pilot and metric evidence.
export R7_RUNS="${R7_RUNS:-/home/dgut/pusht_runtime/r7_review_${R7_TS}}"
if [[ -e "$R7_RUNS/verification.json" ]]; then
  echo "[STOP] output directory already contains verification evidence; use new R7_RUNS" >&2
  exit 3
fi
py="${R7_PYTHON:-/home/dgut/pusht_runtime/venv_oft/bin/python}"
mkdir -p "$R7_RUNS"
echo "[preflight] commit=$(git rev-parse HEAD) mode=$mode python=$py"

if [[ "$mode" == "--check" || "$mode" == "--all" ]]; then
  "$py" tests/test_r6_representable_oracle.py 2>&1 | tail -3
  "$py" tests/test_r7_state_dynamics.py 2>&1 | tail -3
  "$py" diagnostics/r6_representable_oracle.py --out "$R7_RUNS/representable_oracle.json" | tail -20
  "$py" diagnostics/r7_oracle_compare.py --out "$R7_RUNS/oracle_compare.json" | tail -20
  "$py" diagnostics/r6_history_ceiling.py --out "$R7_RUNS/history_ceiling_matched.json" | tail -12
  [[ "$mode" == "--check" ]] && { echo "[check] done (no training launched)"; exit 0; }
fi

if [[ "$mode" == "--mlp" || "$mode" == "--all" ]]; then
  "$py" -u training/r7_state_mlp.py --seeds 0 1 2 --features SV S V --targets R delta \
      --out "$R7_RUNS/state_mlp.json" 2>&1 | tee "$R7_RUNS/mlp_main.log" | tail -20
  "$py" -u training/r7_state_mlp.py --seeds 0 1 2 --features SVAcc SV --targets R --restrict-t2 \
      --out "$R7_RUNS/state_mlp_acc.json" 2>&1 | tee "$R7_RUNS/mlp_acc.log" | tail -20
  [[ "$mode" == "--mlp" ]] && exit 0
fi

"$py" -u diagnostics/r7_dynamics_probe.py --pred-dir "$R7_RUNS" \
    --mlp-json "$R7_RUNS/state_mlp.json" --out "$R7_RUNS/probe_main.json" \
    2>&1 | tee "$R7_RUNS/probe_main.log" | tail -12
"$py" -u diagnostics/r7_dynamics_probe.py --pred-dir "$R7_RUNS" \
    --mlp-json "$R7_RUNS/state_mlp_acc.json" --out "$R7_RUNS/probe_acc.json" \
    2>&1 | tee "$R7_RUNS/probe_acc.log" | tail -12
"$py" diagnostics/r7_verify_claims.py --run-dir "$R7_RUNS" \
    --oracle-json "$R7_RUNS/oracle_compare.json" --mlp-json "$R7_RUNS/state_mlp.json" \
    --acc-json "$R7_RUNS/state_mlp_acc.json" --probe-json "$R7_RUNS/probe_acc.json" \
    --out "$R7_RUNS/verification.json" 2>&1 | tee "$R7_RUNS/verify.log" | tail -20
git rev-parse HEAD > "$R7_RUNS/commit_sha.txt"
echo "[r7] done: artifacts in $R7_RUNS"
