#!/usr/bin/env bash
# R10.2 line A0: read-only snapshot of the old project evaluation + :6020 policy server.
# Nothing here signals, stops or modifies any process.
set -u
OUT=${1:?usage: snapshot.sh OUTDIR}
mkdir -p "$OUT"
PY=/home/dgut/pusht_runtime/venv_oft/bin/python
step() {
  local name=$1; shift
  local log="$OUT/$name.log"
  { echo "### cmd: $*"; echo "### start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; } > "$log"
  "$@" >> "$log" 2>&1
  local rc=$?
  echo "### exit=$rc" >> "$log"
  echo "[$name] exit=$rc"
  return $rc
}

step identity bash -c "date -u +%Y-%m-%dT%H:%M:%SZ; hostname; whoami; id; uname -a; nproc; free -g"
step git bash -c "cd /home/T7/ojh/robot_actual/submission && git rev-parse HEAD && git rev-parse --abbrev-ref HEAD && git show-ref | head -40"
step nvidia_smi nvidia-smi
step nvidia_apps nvidia-smi --query-compute-apps=pid,ppid,process_name,used_memory --format=csv
step gpu_free "$PY" -c "import torch;f,t=torch.cuda.mem_get_info();print('free_gib=%.3f total_gib=%.3f used_gib=%.3f'%(f/2**30,t/2**30,(t-f)/2**30))"
step docker_ps docker ps --format "{{.ID}} {{.Names}} {{.Image}} {{.Status}} {{.Ports}}"
step docker_top docker top rd_setup
step port6020 bash -c "ss -ltnp 2>/dev/null | grep -E ':6020' || echo no-listener-visible-from-host"
step listeners bash -c "ss -ltnp 2>/dev/null | head -20"
step proc_cmdlines bash -c 'for p in $(pgrep -f "eval_client/main.py|_run_client.sh|robodojo.sh client|eval_policy.sh|isaac-sim/kit" 2>/dev/null); do echo "== $p ppid=$(ps -o ppid= -p $p | tr -d " ") user=$(ps -o user= -p $p | tr -d " ")"; tr "\0" " " < /proc/$p/cmdline 2>/dev/null; echo; done'

echo "snapshot -> $OUT"
