#!/bin/bash
# Push-T CLOSED-LOOP DEV run: policy server on the host, OFFICIAL RoboDojo client inside rd_setup.
#
# Why this shape:
#   * the 7B server must run on the HOST (it needs the r11 checkout, the venv and the GPU weights);
#   * the Isaac Sim client must run INSIDE the rd_setup container (network mode host; the container
#     mounts only RoboDojo -> /workspace/RoboDojo, and its python is the one with isaacsim 5.1);
#   * the verdict is never ours: tools/r15_loop_dev.py reads details[].success from the _result.json
#     the official eval client wrote, and enforces the cumulative DEV budget (1 -> 5 -> 20).
# The server is always stopped, on every exit path.
set -uo pipefail
cd /home/T7/ojh/robot_actual/r11
export ROBODOJO_ROOT=/home/T7/ojh/robot_actual/RoboDojo
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
CKPT=${1:?checkpoint dir}
EPISODES=${2:-1}
STAGE=${3:-1}
PORT=${4:-19000}
SRVLOG=logs/r15_policyserver_${PORT}.log
if [ ! -f "$CKPT/MANIFEST.json" ]; then echo "REFUSING: $CKPT has no MANIFEST.json"; exit 9; fi
mkdir -p logs runs/r15_loop_dev
echo "# closed loop $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=$CKPT episodes=$EPISODES stage=$STAGE port=$PORT"
ROBODOJO_ROOT=$ROBODOJO_ROOT PYTHONPATH=$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab   "$PY" code/xpolicylab/policy/OpenVLA_R42/r42_serve.py \
  --config_path code/xpolicylab/policy/OpenVLA_R42/deploy.yml \
  --host 0.0.0.0 --port "$PORT" --overrides checkpoint_dir="$CKPT" > "$SRVLOG" 2>&1 &
SRV=$!
cleanup() { kill -TERM "$SRV" 2>/dev/null; sleep 3; kill -KILL "$SRV" 2>/dev/null; }
trap cleanup EXIT INT TERM
READY=0
for i in $(seq 1 240); do
  if timeout 2 bash -c ">/dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "POLICY SERVER DIED during warmup; last 40 lines:"; tail -n 40 "$SRVLOG"; exit 8
  fi
  sleep 5
done
if [ "$READY" != 1 ]; then echo "POLICY SERVER NOT READY after 1200s"; tail -n 40 "$SRVLOG"; exit 8; fi
echo "policy server ready on ws://0.0.0.0:$PORT (pid $SRV) after $((i*5))s"
grep -a "OpenVLA_R42\|codec\|metadata" "$SRVLOG" | tail -8
"$PY" tools/r15_loop_dev.py --task push_T --policy-name OpenVLA_R42 --port "$PORT" \
  --episodes "$EPISODES" --stage "$STAGE" --docker rd_setup --ckpt-label "$(basename "$CKPT")" \
  --timeout-s 3000 \
  --out "runs/r15_loop_dev/push_T_stage${STAGE}_$(date -u +%Y%m%dT%H%M%SZ).json"
RC=$?
echo "# closed loop finished rc=$RC at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $RC
