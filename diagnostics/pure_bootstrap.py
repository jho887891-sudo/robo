#!/usr/bin/env python3
"""纯 OpenVLA 离线诊断的公共引导层：加载部署同款模型、构造固定留出样本。

只做三件事，避免每个诊断脚本再抄一遍环境装配：
  1. bootstrap()  —— 与 RoboDojo 部署完全相同的 import 路径与参数
  2. load_model() —— 直接用部署类 XPolicyLab.policy.OpenVLA_Pure.model.Model，
                     因此适配器 / proprio 投影层 / q01-q99 / BOUNDS 统计参数
                     与线上服务是同一套代码加载的
  3. iter_samples() —— 固定留出样本（fixed_samples.json 前 N 个），三路相机 + 14 维状态

不训练、不改检查点、不连仿真。
"""
import json
import os
import sys

import numpy as np

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
ROBO = os.environ.get("PUSHT_ROBODOJO", "/home/T7/ojh/robot_actual/RoboDojo")
OFT = os.environ.get("PUSHT_OFT", ROBO + "/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")
HDF5_DIR = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
DEPLOY_YML = os.environ.get("PUSHT_DEPLOY_YML", ROBO + "/XPolicyLab/policy/OpenVLA_Pure/deploy.yml")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")
SAMPLES = os.environ.get("PUSHT_SAMPLES", RT + "/fixed_samples.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def bootstrap():
    """装好 import 路径，并吃掉 prismatic 需要的 argv 参数。"""
    for p in (OFT, ROBO):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "--aloha" not in sys.argv:
        sys.argv = [sys.argv[0], "--aloha"]


def deploy_cfg(task="push_T", env_cfg_type="arx_x5", action_type="joint", **over):
    import yaml
    cfg = yaml.safe_load(open(DEPLOY_YML))
    cfg.update({"task_name": task, "env_cfg_type": env_cfg_type, "action_type": action_type})
    cfg.update(over)
    return cfg


def load_model(task="push_T", **over):
    bootstrap()
    import XPolicyLab.policy.OpenVLA_Pure.model as M
    m = M.Model(deploy_cfg(task=task, **over))
    return m


def bounds():
    nb = np.load(BOUNDS)
    return np.asarray(nb["amin"], dtype=np.float32), np.asarray(nb["amax"], dtype=np.float32)


def norm_bounds(a, amin, amax):
    """BOUNDS 归一化（与 OFT 训练/评测同一套 amin/amax）。"""
    return np.clip(2.0 * (np.asarray(a, np.float32) - amin) / (amax - amin + 1e-8) - 1.0, -1, 1).astype(np.float32)


def c14(h, g):
    return np.concatenate([
        h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
        h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:],
    ], axis=1).astype(np.float32)


def decode_img(buf):
    import cv2
    return cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()


def iter_samples(n=20, samples_path=None):
    """固定留出样本：返回 [(episode, t, images_rgb(list of 3), state14, action14)]。"""
    import h5py
    path = samples_path or SAMPLES
    fixed = json.load(open(path))[:n]
    out = []
    for e, t in fixed:
        with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
            imgs = [decode_img(h["vision/" + c + "/colors"][t]) for c in CAMS]
            state = c14(h, "state")[t].copy()
            action = c14(h, "action")[t].copy()
        out.append((e, t, imgs, state, action))
    return out


def make_obs(imgs, state, instruction):
    return {
        "images": {"cam_high": imgs[0], "cam_left_wrist": imgs[1], "cam_right_wrist": imgs[2]},
        "state": np.asarray(state, dtype=np.float32),
        "instruction": instruction,
    }


DIM_NAMES = (["left_arm_j%d" % i for i in range(6)] + ["left_gripper"] +
             ["right_arm_j%d" % i for i in range(6)] + ["right_gripper"])
