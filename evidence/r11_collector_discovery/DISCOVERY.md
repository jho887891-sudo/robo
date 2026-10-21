# R11 collector — read-only expert discovery (result: BLOCKED_EXPERT)

Date: 2026-10-20 19:52Z–19:53Z. GPU seconds spent: 0. No episode was attempted.

## What was inspected (read-only)

```
docker exec rd_setup bash -lc "ls /workspace/RoboDojo"           # nvcr.io/nvidia/isaac-sim:5.1.0
docker exec rd_setup bash -lc "ls /workspace/RoboDojo/scripts"
docker exec rd_setup bash -lc "grep -n 'client|policy-name|expert' /workspace/RoboDojo/scripts/robodojo.sh"
docker exec rd_setup bash -lc "find /workspace -maxdepth 5 -iname '*expert*'"
docker exec rd_setup bash -lc "ls /workspace/RoboDojo/XPolicyLab/policy"
```

## Findings

1. `scripts/robodojo.sh` exposes `eval` / `server` / `client`. `--policy-name NAME` resolves to
   `XPolicyLab/policy/<NAME>/deploy.py`, i.e. the policy **under test**. There is no `--expert`
   flag and no shipped expert controller for `push_T`.
2. `XPolicyLab/policy/` contains model policy servers only (A1, ACT, ACT14, DP, GR00T_N17,
   OpenVLA variants, ...). The single hit for "expert" in the whole container is
   `XPolicyLab/policy/openvla_flat3/deploy_pybullet_expert_backup.py`, which belongs to the
   pybullet variant and cannot drive the Isaac Sim `push_T` task.
3. The existing 100 Push-T episodes were **downloaded** (`scripts/RoboDojo/download_data.sh`),
   not generated on this machine, so the generator that produced them is not available here.
4. `code/robodojo/push_T_progress.py` in this repo proves the sim client can expose
   `t`/`target_t` poses and end-effector poses per control step — enough for a geometric expert,
   but that expert does not exist yet.

## Decision

No expert trajectory source is executable and auditable on this workstation, therefore:

* COLLECTOR bucket: **0 s spent** (14400 s available),
* no HDF5 was written, no NEW_DEV / NEW_FINAL split exists,
* `BLOCKED_EXPERT` recorded in the mission manifest.

Fabricating trajectories from random or all-zero actions is explicitly forbidden by the spec and
was not done.

## Unblock plan (in order of cost)

1. Implement a scripted geometric Push-T expert on top of the RoboDojo env API (privileged
   `t`/`target_t` poses -> push direction -> IK waypoints), pre-register the seed list and the
   acceptance rule **before** the first attempt, validate success on a small DEV-like set, then
   run the 40-attempt / 25-valid collection inside the 4 h COLLECTOR bucket.
2. Only then freeze 5 NEW_DEV + 20 NEW_FINAL through `tools/r10_freeze_guard.py` and push the
   freeze commit before any model sees them.
