#!/usr/bin/env bash
set -u
: "${R46_CPU_WORKTREE:?set R46_CPU_WORKTREE to the clean pinned worktree}"
: "${R46_CPU_PYTHON:?set R46_CPU_PYTHON to the verified CPU interpreter}"
cd "$R46_CPU_WORKTREE"
E=/tmp/r46-cpu-gate-final-20261029T$(date -u +%H%M%S)Z
mkdir -m 700 "$E"
POINTER=${R46_CPU_EVIDENCE_POINTER:-/tmp/r46-cpu-gate-final-evidence-dir}
printf '%s\n' "$E" > "$POINTER"
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
PY=$R46_CPU_PYTHON
{
  echo "git_head=$(git rev-parse HEAD)"
  echo "git_status=$(git status --porcelain=v2 --branch | tr '\n' ';')"
  "$PY" -c 'import sys,numpy,torch,transformers,peft,h5py,cv2; print("python",sys.version.replace("\\n"," ")); print("numpy",numpy.__version__); print("torch",torch.__version__); print("transformers",transformers.__version__); print("peft",peft.__version__); print("h5py",h5py.__version__); print("opencv",cv2.__version__)'
} > "$E/environment.txt" 2> "$E/environment.stderr"
sha256sum diagnostics/r46_smoke_supervisor.py diagnostics/r46_smoke_contract.py diagnostics/r46_openvla_smoke.py diagnostics/r46_freeze_samples.py tests/test_r46_smoke_supervisor.py tests/test_r46_smoke_contract.py tests/test_r46_freeze_samples.py config/r46_smoke_samples.json > "$E/source_SHA256SUMS"
run_case() {
  name=$1; shift
  set +e
  "$PY" "$@" > "$E/$name.stdout" 2> "$E/$name.stderr"
  rc=$?
  set -u
  printf '%s\n' "$rc" > "$E/$name.exit_code"
  printf '%s exit=%s\n' "$name" "$rc"
}
run_case ast_parse -c 'import ast,pathlib; files=["training/r44_train.py","training/r45_data.py","training/r45_train.py","diagnostics/r46_smoke_supervisor.py","diagnostics/r46_smoke_contract.py","diagnostics/r46_openvla_smoke.py","diagnostics/r46_freeze_samples.py"]; [ast.parse(pathlib.Path(f).read_text()) for f in files]; print("AST_PARSE_PASS", len(files))'
run_case py_compile -m py_compile diagnostics/r46_smoke_supervisor.py diagnostics/r46_smoke_contract.py diagnostics/r46_openvla_smoke.py diagnostics/r46_freeze_samples.py tests/test_r46_smoke_supervisor.py tests/test_r46_smoke_contract.py tests/test_r46_freeze_samples.py tests/test_r46_openvla_smoke_args.py
run_case r46_contract -m unittest discover -s tests -p test_r46_smoke_contract.py -v
run_case r46_freeze_tests -m unittest discover -s tests -p test_r46_freeze_samples.py -v
run_case r46_supervisor -m unittest discover -s tests -p test_r46_smoke_supervisor.py -v
run_case r46_openvla_args -m unittest discover -s tests -p test_r46_openvla_smoke_args.py -v
run_case r45_runtime_gate -m unittest discover -s tests -p test_r45_runtime_gate.py -v
run_case r45_toy_resume -m unittest discover -s tests -p test_r45_toy_resume.py -v
run_case r44_delta tests/test_r44_delta_v1.py
run_case r45_contract tests/test_r45_contract.py
run_case r44_train_core -m unittest discover -s tests -p test_r44_train_core.py -v
run_case r45_pipeline -m unittest discover -s tests -p test_r45_pipeline.py -v
failed=0
for f in "$E"/*.exit_code; do
  rc=$(cat "$f")
  [ "$rc" = 0 ] || failed=1
done
sha256sum "$E"/*.stdout "$E"/*.stderr "$E"/*.exit_code > "$E/SHA256SUMS"
cat "$E"/*.exit_code
printf 'EVIDENCE_DIR=%s\n' "$E"
exit "$failed"
