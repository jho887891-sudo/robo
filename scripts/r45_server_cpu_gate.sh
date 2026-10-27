#!/usr/bin/env bash
# R45-F: CPU-only host preflight with immutable evidence and no CUDA initialization.
# Execute only from a clean independent worktree at an explicitly pinned commit.
set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="none"
export PYTHONDONTWRITEBYTECODE=1

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

fail() { printf 'BLOCKED: %s\n' "$*" >&2; exit 2; }
: "${R45_EXPECTED_SHA:?Set R45_EXPECTED_SHA to the reviewed full commit SHA}"
for name in M2_HDF5_ROOT M2_NORMALIZER_JSON R44_DELTA_NORMALIZER R44_DATASET_MANIFEST R44_BASE_MODEL R44_OFT_ROOT; do
  [[ -n "${!name:-}" ]] || fail "Missing server environment variable: $name"
done
[[ "$(git rev-parse HEAD)" == "$R45_EXPECTED_SHA" ]] || fail "Git HEAD differs from reviewed SHA"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "worktree is dirty"
[[ -f "$R44_BASE_MODEL/config.json" ]] || fail "model config missing"
[[ -d "$M2_HDF5_ROOT" ]] || fail "HDF5 root missing"
for p in "$M2_NORMALIZER_JSON" "$R44_DELTA_NORMALIZER" "$R44_DATASET_MANIFEST"; do
  [[ -f "$p" ]] || fail "CPU contract file missing: $p"
done
[[ -d "$R44_OFT_ROOT" ]] || fail "OFT source tree missing"

evroot="${R45_EVIDENCE_ROOT:-/tmp}"
[[ -d "$evroot" ]] || fail "evidence parent does not exist: $evroot"
out="$(mktemp -d "$evroot/r45-f-cpu-XXXXXXXX")"
printf 'R45_F_EVIDENCE=%s\n' "$out"
printf '%s\n' "$R45_EXPECTED_SHA" > "$out/code_sha.txt"
python3 --version > "$out/python_version.txt" 2>&1
git status --porcelain=v2 --branch > "$out/git_status.txt"
git remote -v > "$out/git_remotes.txt"
git rev-parse HEAD > "$out/git_head.txt"

# Import/version queries on CPU only; never touch torch.cuda or nvidia-smi here.
python3 - > "$out/packages.json" 2> "$out/packages.stderr" <<'PY'
import importlib.metadata as md
import json, platform
packages = ["torch", "h5py", "numpy", "transformers", "peft"]
print(json.dumps({
    "python": platform.python_version(),
    "packages": {name: md.version(name) for name in packages},
}, indent=2, sort_keys=True))
PY

set +e
python3 -m training.r44_train \
  --preflight-only --out "$out/preflight-output-unused" \
  --splits "$root/data/m2_splits.json" \
  --hdf5-root "$M2_HDF5_ROOT" \
  --feature-normalizer "$M2_NORMALIZER_JSON" \
  --delta-normalizer "$R44_DELTA_NORMALIZER" \
  --dataset-manifest "$R44_DATASET_MANIFEST" \
  --base-model "$R44_BASE_MODEL" \
  --oft-root "$R44_OFT_ROOT" > "$out/r44_real_data_preflight.stdout" 2> "$out/r44_real_data_preflight.stderr"
preflight_rc=$?
printf 'r44_real_data_preflight=%s\n' "$preflight_rc" > "$out/exit_codes.txt"

python3 -m unittest discover -s tests -p 'test_r45_runtime_gate.py' -v \
  > "$out/r45_runtime.stdout" 2> "$out/r45_runtime.stderr"
runtime_rc=$?
printf 'r45_runtime=%s\n' "$runtime_rc" >> "$out/exit_codes.txt"

python3 -m unittest discover -s tests -p 'test_r45_toy_resume.py' -v \
  > "$out/r45_toy_resume.stdout" 2> "$out/r45_toy_resume.stderr"
toy_rc=$?
printf 'r45_toy_resume=%s\n' "$toy_rc" >> "$out/exit_codes.txt"

python3 tests/test_r44_delta_v1.py > "$out/r44_delta.stdout" 2> "$out/r44_delta.stderr"
delta_rc=$?
printf 'r44_delta=%s\n' "$delta_rc" >> "$out/exit_codes.txt"

python3 tests/test_r45_contract.py > "$out/r45_contract.stdout" 2> "$out/r45_contract.stderr"
contract_rc=$?
printf 'r45_contract=%s\n' "$contract_rc" >> "$out/exit_codes.txt"
set -e

(
  cd "$out"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 |
    sort -z | xargs -0 -r sha256sum > SHA256SUMS
)
if (( preflight_rc || runtime_rc || toy_rc || delta_rc || contract_rc )); then
  printf 'STATUS=BLOCKED_CPU_GATE\nEVIDENCE=%s\n' "$out"
  exit 3
fi
printf 'STATUS=PASS_CPU_GATE_ONLY\nEVIDENCE=%s\n' "$out"
printf 'GPU_SMOKE=NOT_AUTHORIZED\nGPU_BENCHMARK=NOT_IMPLEMENTED\n'
