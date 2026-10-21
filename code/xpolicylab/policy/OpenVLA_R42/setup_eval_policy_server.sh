#!/usr/bin/env bash
# R16: OpenVLA_R42 server launcher in the RoboDojo convention.
#
# Called by scripts/robodojo.sh (eval/server mode) as
#   bash setup_eval_policy_server.sh BENCH TASK CKPT ENV_CFG ACTION_TYPE SEED GPU_ID ENV PORT [HOST]
# and it also works standalone:
#   bash setup_eval_policy_server.sh "" push_T "" arx_x5 joint 0 0 "" 19000 0.0.0.0
#
# Differences from the stock launcher (XPolicyLab/policy/demo_policy/setup_eval_policy_server.sh):
#   * no conda: the 42D runtime lives in a venv (R42_PYTHON, default below);
#   * the model is served straight from this r11 directory (r42_serve.py), so nothing has to be
#     copied into the RoboDojo tree for split mode (robodojo.sh client + this server);
#   * the served codec mode and the metadata provenance are printed by the model itself (G12).
set -euo pipefail

bench_name=${1:-}
task_name=${2:-push_T}
ckpt_name=${3:-}
env_cfg_type=${4:-arx_x5}
action_type=${5:-joint}
seed=${6:-0}
policy_gpu_id=${7:-0}
policy_conda_env=${8:-}
policy_server_port=${9:-19000}
policy_server_host=${10:-0.0.0.0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R11_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ROBODOJO_ROOT=${ROBODOJO_ROOT:-/home/T7/ojh/robot_actual/RoboDojo}
R42_PYTHON=${R42_PYTHON:-/home/dgut/pusht_runtime/venv_oft/bin/python}

export PYTHONPATH="${ROBODOJO_ROOT}:${ROBODOJO_ROOT}/XPolicyLab:${R11_ROOT}/code/xpolicylab/policy:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore::UserWarning
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
export PYTHONUNBUFFERED=1

overrides=(
  policy_name=OpenVLA_R42
  task_name="${task_name}"
  bench_name="${bench_name}"
  action_type="${action_type}"
  env_cfg_type="${env_cfg_type}"
  seed="${seed}"
  host="${policy_server_host}"
  port="${policy_server_port}"
)
# ckpt_name is "the checkpoint to serve": a stage checkpoint directory when it is a path.
if [[ -n "${ckpt_name}" && -d "${ckpt_name}" ]]; then
  overrides+=(checkpoint_dir="${ckpt_name}")
fi

echo "[R42-SERVER] policy=OpenVLA_R42 task=${task_name} ckpt=${ckpt_name:-<deploy.yml>}"
echo "[R42-SERVER] python=${R42_PYTHON} gpu=${policy_gpu_id} port=${policy_server_port} host=${policy_server_host}"
echo "[R42-SERVER] config=${SCRIPT_DIR}/deploy.yml"

exec "${R42_PYTHON}" "${SCRIPT_DIR}/r42_serve.py" \
  --config_path "${SCRIPT_DIR}/deploy.yml" \
  --overrides "${overrides[@]}"
