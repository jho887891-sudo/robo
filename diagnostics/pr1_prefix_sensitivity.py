#!/usr/bin/env python3
"""第三轮 PR 验证 · 附加实验：前序动作前缀敏感性（回答“修复是否恢复了自回归条件依赖”）。

同一模型、同一图像/指令/本体状态，只替换 14 个前序动作 token：

  zero 机制（OFT 原行为）：动作输入被置零 -> 预测**必须**完全不变（条件通路被切断）
  fix  机制（PR 的 enable_pure_autoreg）：预测应当随前序动作改变（条件通路存在）

用法：python diagnostics/pr1_prefix_sensitivity.py --out /tmp/pr1_prefix.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
META = os.environ.get("PUSHT_META", RT + "/model_cache/pure3c/run_meta.json")
MODEL = os.environ.get("PUSHT_MODEL_PRISTINE", RT + "/model_cache/openvla-7b-pristine")
ADAPTER = os.environ.get("PUSHT_ADAPTER", RT + "/model_cache/pure3c/lora_adapter")
PPF = os.environ.get("PUSHT_PROJ", RT + "/model_cache/pure3c/proprio_projector--pure.pt")
OFT = os.environ.get("PUSHT_OFT", "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")
CODE = os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure")

sys.path.insert(0, CODE); sys.path.insert(0, OFT)
# 只追加 --aloha，不要替换 sys.argv（否则 --out 等命令行参数会被吞掉）
if "--aloha" not in sys.argv:
    sys.argv.append("--aloha")

import action_codec as AC
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector

DEV = "cuda"; W0 = AC.LEGAL_TOKEN_MIN
INSTR = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
DIMS = (["left_arm_j%d" % i for i in range(6)] + ["left_gripper"] +
        ["right_arm_j%d" % i for i in range(6)] + ["right_gripper"])
PASS, FAIL = "PASS", "FAIL"


def c14(h, g):
    return np.concatenate([h[g + "/left_arm_joint_states"][:], h[g + "/left_ee_joint_states"][:],
                           h[g + "/right_arm_joint_states"][:], h[g + "/right_ee_joint_states"][:]],
                          axis=1).astype(np.float32)


def dec(b):
    import cv2
    return cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args, _u = ap.parse_known_args()

    spec = importlib.util.spec_from_file_location("pure_autoreg_mode",
                                                  os.path.join(CODE, "pure_autoreg_mode.py"))
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)

    import h5py
    meta = json.load(open(META)); q01 = np.asarray(meta["q01"], np.float32); q99 = np.asarray(meta["q99"], np.float32)
    nb = np.load(BOUNDS); pmin = np.asarray(nb["pmin"], np.float32); pmax = np.asarray(nb["pmax"], np.float32)
    e0 = sorted(json.load(open(SPLIT))["val"])[0]
    with h5py.File(os.path.join(HDF5, e0), "r") as h:
        t = 20
        imgs = [dec(h["vision/" + c + "/colors"][t]) for c in CAMS]
        state = c14(h, "state")[t].copy(); act = c14(h, "action")[t].copy()

    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True, trust_remote_code=True).to(DEV)
    vla.vision_backbone.set_num_images_in_input(3)
    orig_method = vla._process_action_masks
    vla = PeftModel.from_pretrained(vla, ADAPTER, is_trainable=False).merge_and_unload()
    pp = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim)
    pp.load_state_dict(torch.load(PPF, map_location="cpu")); pp = pp.to(torch.bfloat16).to(DEV)
    vla.eval()

    pv = torch.cat([proc.image_processor.apply_transform(Image.fromarray(im)) for im in imgs],
                   dim=0).unsqueeze(0).to(torch.bfloat16).to(DEV)
    pn = np.clip(2.0 * (state - pmin) / (pmax - pmin + 1e-8) - 1.0, -1, 1).astype(np.float32)
    proprio = torch.from_numpy(pn).unsqueeze(0).to(DEV)
    b = PurePromptBuilder("openvla")
    b.add_turn("human", "What action should the robot take to " + INSTR + "?")
    pids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to(DEV)
    if int(pids[0, -1].item()) != 29871:
        pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype, device=DEV)], dim=1)
    n_text = int(pids.shape[1])

    gt_idx = AC.encode_indices(AC.normalize(act, q01, q99), mode=AC.MODE_FIXED)
    gt_tok = [int(x) for x in AC.token_ids(gt_idx)]
    alt_tok = [int(W0 + (0 if i % 2 == 0 else 255)) for i in range(14)]

    def predict(tokens, mode):
        if mode == "zero":
            vla._process_action_masks = orig_method
        else:
            helper.enable_pure_autoreg(vla)
        aid = torch.tensor([tokens], dtype=torch.long, device=DEV)
        stop = torch.full((1, 1), AC.STOP_INDEX, dtype=torch.long, device=DEV)
        inp = torch.cat([pids, aid, stop], dim=1)
        lab = torch.cat([torch.full_like(pids, -100), aid, stop], dim=1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    labels=lab, proprio=proprio, proprio_projector=pp)
        n_vis = int(o.logits.shape[1]) - int(inp.shape[1])
        idxs = []
        for k in range(14):
            lg = o.logits[0, n_vis + n_text + k - 1]
            idxs.append(int(torch.argmax(lg[W0:W0 + AC.N_BINS]).item()))
        idxs = np.asarray(idxs)
        return idxs, AC.actions_from_token_ids(AC.token_ids(idxs), q01, q99)

    res = {"episode": e0, "frame": t, "dims": DIMS, "checks": {}}
    for mode in ("zero", "fix"):
        i_gt, a_gt = predict(gt_tok, mode)
        i_alt, a_alt = predict(alt_tok, mode)
        d = np.abs(a_gt - a_alt)
        res[mode] = {"tokens_gt": i_gt.tolist(), "tokens_alt": i_alt.tolist(),
                     "tokens_equal": int((i_gt == i_alt).sum()),
                     "action_l1": float(d.mean()), "action_max": float(d.max()),
                     "per_dim_l1": d.tolist()}
        print("[%s] GT-prefix vs swapped-prefix: tokens equal %d/14 | action L1 = %.6f (max %.6f)" % (
            mode.upper(), res[mode]["tokens_equal"], res[mode]["action_l1"], res[mode]["action_max"]), flush=True)
        print("      per-dim L1: %s" % [round(float(x), 5) for x in d], flush=True)

    ck = res["checks"]
    ck["zero_mode_ignores_prefix"] = res["zero"]["tokens_equal"] == 14 and res["zero"]["action_max"] == 0.0
    ck["fix_mode_uses_prefix"] = res["fix"]["action_max"] > 1e-4
    res["passed"] = all(ck.values())
    print("\n=== RESULT ===")
    for k, v in ck.items():
        print("  %-34s %s" % (k, PASS if v else FAIL))
    print("PR1_PREFIX_SENSITIVITY: %s" % ("ALL_PASS" if res["passed"] else "FAILED"))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print("wrote " + args.out)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
