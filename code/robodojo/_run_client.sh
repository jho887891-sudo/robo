#!/bin/bash
# usage: _run_client.sh <task> <policy_name> <port> <eval_num>
set -u
TASK=${1:-push_T}; POLICY=${2:-OpenVLA_Pure}; PORT=${3:-6020}; NUM=${4:-1}
export CUROBO_KERNEL_BACKEND=cuda_core
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
cd /workspace/RoboDojo || exit 1
echo "[runner] task=$TASK policy=$POLICY port=$PORT num=$NUM labels=${ROBODOJO_PROGRESS_LABELS:-default}"
bash scripts/robodojo.sh client --task "$TASK" --policy-name "$POLICY" \
  --policy-host 127.0.0.1 --policy-port "$PORT" --eval-num "$NUM"
