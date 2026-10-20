#!/usr/bin/env bash
# Gate A: refit the frozen 42D normalisation + target-R contract on 64 TRAIN only.
# Every command, exit code and log is archived. No GPU, no optimizer step.
set -u
cd /home/T7/ojh/robot_actual/r9_A
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
export PUSHT_HDF5=${PUSHT_HDF5:-/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data}
export PUSHT_BOUNDS=${PUSHT_BOUNDS:-/home/dgut/pusht_runtime/push_T_bounds.npz}
export R9_OUT=${R9_OUT:-/home/dgut/pusht_runtime/r9_gate_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$R9_OUT/logs"
TSV="$R9_OUT/commands.tsv"
: > "$TSV"

step() {
  local name=$1; shift
  local log="$R9_OUT/logs/$name.log"
  printf "%s\t%s\t" "$name" "$*" >> "$TSV"
  { echo "### step=$name"; echo "### cwd=$(pwd)"; echo "### cmd: $*";
    echo "### start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$log"
  "$@" >> "$log" 2>&1
  local rc=$?
  echo "### exit=$rc" >> "$log"
  echo "$rc" >> "$TSV"
  echo "[$name] exit=$rc"
  return $rc
}

step env_snapshot "$PY" tools/r9_env_snapshot.py "$R9_OUT/environment.json"
step py_compile "$PY" -m py_compile tools/make_frozen_contract.py diagnostics/r8_smoke.py diagnostics/r6_data.py training/r7_state_mlp.py
step test_meta_audit "$PY" tests/test_r8_frozen_metadata_audit.py
step test_normalization "$PY" tests/test_r8_normalization.py
step test_vla_adapter "$PY" tests/test_r8_vla_adapter.py
step test_contract "$PY" tests/test_r8_contract.py
step sha_before_old80 sha256sum evidence/r8_vla_42d_smoke_20261020/frozen_r8_contract.json

step freeze_64train "$PY" tools/make_frozen_contract.py \
  --contracts evidence/r8_vla_42d_smoke_20261020/frozen_norm_contracts.json \
  --scheme B99_signed --episodes 0 \
  --out "$R9_OUT/frozen_contract_64train.json"
step cpu_check_contract "$PY" diagnostics/r8_smoke.py --check-contract \
  --norm-contract "$R9_OUT/frozen_contract_64train.json" --out "$R9_OUT/cpu_contract"

# Fail-closed: the previous 80-episode artefact must be REJECTED on the new path.
step fail_closed_old80 "$PY" diagnostics/r8_smoke.py --check-contract \
  --norm-contract evidence/r8_vla_42d_smoke_20261020/frozen_r8_contract.json \
  --out "$R9_OUT/cpu_old80_rejected"
OLD_RC=$(awk -F"\t" '$1=="fail_closed_old80"{print $3}' "$TSV" | tail -1)
step sha_after_old80 sha256sum evidence/r8_vla_42d_smoke_20261020/frozen_r8_contract.json

FAILED=0
if [ "$OLD_RC" = "0" ]; then echo "FATAL: old 80-episode contract was accepted"; FAILED=1; fi
for s in env_snapshot py_compile test_meta_audit test_normalization test_vla_adapter test_contract freeze_64train cpu_check_contract; do
  rc=$(awk -F"\t" -v n="$s" '$1==n{print $3}' "$TSV" | tail -1)
  if [ "$rc" != "0" ]; then echo "NOTE: step $s exit=$rc"; fi
done
echo "R9_OUT=$R9_OUT"
echo "FAILED=$FAILED"
exit $FAILED
