"""Process-progress logger for RoboDojo tasks.

Binary success gives no gradient when every policy fails, so record raw geometry
every few control steps: object poses, end-effector poses and the derived errors.

    PUSHT_PROGRESS_EVERY      -- log every N control steps (default 5)
    ROBODOJO_PROGRESS_LABELS  -- comma separated object labels to log
                                 (default "t,target_t" for push_T)

Rows land in eval_result/_progress/progress_<run_id>.jsonl
"""
import json
import os
import time

import numpy as np

_STATE = {
    "step_last": -1,
    "episode": 0,
    "path": None,
    "diag_done": False,
    "err_done": False,
}
_EVERY = int(os.environ.get("PUSHT_PROGRESS_EVERY", "5"))
_LABELS = [s for s in os.environ.get("ROBODOJO_PROGRESS_LABELS", "t,target_t").split(",") if s]


def _progress_dir():
    base = os.path.join(os.getcwd(), "eval_result", "_progress")
    try:
        os.makedirs(base, exist_ok=True)
        return base
    except Exception:
        return "/tmp"


def _path():
    if _STATE["path"] is None:
        run_id = os.environ.get("ROBODOJO_RUN_ID", "run")
        _STATE["path"] = os.path.join(_progress_dir(), "progress_%s.jsonl" % run_id)
    return _STATE["path"]


def _diag(msg):
    try:
        with open(os.path.join(_progress_dir(), "logger_diag.txt"), "a") as fh:
            fh.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    except Exception:
        return


def _quat_angle_deg(qa, qb):
    try:
        from env.reward_manager.func_parser import cal_quat_dis

        return float(cal_quat_dis(qa, qb) * 180.0 / np.pi)
    except Exception:
        return None


def log_progress(task_env):
    try:
        rm = getattr(task_env, "reward_manager", None)
        fp = getattr(rm, "func_parser", None)
        lm = getattr(fp, "layout_manager", None)
        counts = getattr(task_env, "take_action_cnt", None)

        if not _STATE["diag_done"]:
            _STATE["diag_done"] = True
            _diag("first call: rm=%s fp=%s lm=%s cnt=%s labels=%s" % (
                type(rm).__name__, type(fp).__name__, type(lm).__name__, counts, _LABELS))

        if lm is None or counts is None:
            return

        step = int(np.asarray(counts).reshape(-1)[0])
        if _EVERY > 1 and (step % _EVERY) != 0:
            return
        if step <= _STATE["step_last"]:
            _STATE["episode"] += 1
        _STATE["step_last"] = step

        objs = {}
        for label in _LABELS:
            try:
                name = lm.get_instance_name(label=label, env_idx=0)
                if name is None:
                    continue
                pos, quat = lm.get_instance_pose(inst_name=name, env_idx=0)
                objs[label] = {
                    "pos": [round(float(v), 5) for v in np.asarray(pos, dtype=float).reshape(-1)[:3]],
                    "quat": [round(float(v), 5) for v in np.asarray(quat, dtype=float).reshape(-1)[:4]],
                }
            except Exception:
                continue

        rec = {
            "wall": round(time.time(), 2),
            "ep": _STATE["episode"],
            "step": step,
        }
        try:
            seed_map = getattr(task_env, "current_env_seed_map", None)
            if seed_map:
                rec["seed"] = int(seed_map.get(0, -1))
        except Exception:
            pass

        # primary pair geometry (push_T: block "t" vs pad "target_t")
        if len(_LABELS) >= 2 and _LABELS[0] in objs and _LABELS[1] in objs:
            a = objs[_LABELS[0]]
            b = objs[_LABELS[1]]
            pa = np.asarray(a["pos"], dtype=float)
            pb = np.asarray(b["pos"], dtype=float)
            rec["xy"] = round(float(np.linalg.norm(pa[:2] - pb[:2])), 6)
            rec["dz"] = round(float(pa[2] - pb[2]), 4)
            ang = _quat_angle_deg(a["quat"], b["quat"])
            if ang is not None:
                rec["ang_deg"] = round(ang, 3)

        # end-effector poses + EE-to-first-object distance
        eef = {}
        try:
            roboman = getattr(fp, "robot_manager", None)
            if roboman is not None and hasattr(roboman, "get_real_endpose"):
                for idx_robot, robot in enumerate(getattr(roboman, "robot_list", [])):
                    if getattr(robot, "type", None) != "target":
                        continue
                    pose = roboman.get_real_endpose(robot, env_idx_list=[0])[0]
                    arr = np.asarray(pose, dtype=float).reshape(-1)
                    eef[str(idx_robot)] = [round(float(v), 5) for v in arr[:3]]
                    if _LABELS and _LABELS[0] in objs and arr.size >= 3:
                        target = np.asarray(objs[_LABELS[0]]["pos"], dtype=float)
                        rec["ee_dist_%s_%s" % (idx_robot, _LABELS[0])] = round(
                            float(np.linalg.norm(arr[:3] - target)), 5)
        except Exception:
            pass
        if eef:
            rec["eef"] = eef
        if objs:
            rec["obj"] = objs

        with open(_path(), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as exc:
        if not _STATE.get("err_done"):
            _STATE["err_done"] = True
            _diag("EXC %r" % (exc,))
        return
