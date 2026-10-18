#!/usr/bin/env python3
"""第三轮：纯 OpenVLA 自回归动作输入掩码探针（先取证，不改模型）。

背景
----
code/openvla_dynamic_modules/modeling_prismatic.py 的多模态前向里：

    all_actions_mask = self._process_action_masks(labels)      # 由 labels 推出 14 个动作位置
    ...
    input_embeddings = input_embeddings * ~all_actions_mask      # 动作位置输入嵌入被置零

而缓存式生成分支（input_ids.shape[1] == 1）assert labels is None，直接把真实 token 送进
language_model —— 动作位置拿到的是**真实嵌入**。

本脚本用同一帧真实样本，量化“训练机制（掩码/并行）”与“部署机制（真实动作标记/自回归）”的差异：

  A  = 全序列 + labels  （= 训练与训练内 evaluate 的机制，动作输入被置零）
  B  = 全序列 + labels，但把 _process_action_masks 临时替换成全 False
       （= 关闭置零后的自回归机制，等价于本轮修复后的训练机制；不改文件即可测）
  C  = 部署路径 model.Model.infer（KV 缓存逐 token 生成，labels=None → 真实嵌入）
  D  = 探针自写的逐 token 生成（同一机制但独立实现，用于验证 C 的实现）

同时打印：
  1. 14 个动作位置的 all_actions_mask 布尔值；
  2. 进入 language_model 之前的动作位置输入嵌入是否全零（前向 hook）；
  3. A vs B 的 14 个位置的窗口概率 / 标记 / 动作差；
  4. C vs D 的一致性；A 机制下的自由运行结果 vs C；
  5. 因果性：LM 注意力严格上三角是否为 0 + 篡改未来动作标记是否影响过去位置。

用法：
  python diagnostics/r3_mask_probe.py --out /tmp/r3_probe.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pure_bootstrap as B

B.bootstrap()
from XPolicyLab.policy.OpenVLA_Pure import action_codec as AC

DEV = "cuda"
W0 = AC.LEGAL_TOKEN_MIN                     # 31744
PROMPT_IGNORE = -100


def window_probs(logits):
    cand = logits[W0: W0 + AC.N_BINS].float()
    p = torch.softmax(cand, dim=-1)
    top = int(torch.argmax(p).item())
    t3 = torch.topk(p, 3)
    return {
        "idx": top,
        "p": float(p[top].item()),
        "top3": [[int(i), float(v)] for i, v in zip(t3.indices.tolist(), t3.values.tolist())],
        "probs": p.detach().cpu().numpy(),
    }


def masked_off(vla):
    """返回一个把动作位置掩码强制为全 False 的补丁（等价于“不置零”）。"""
    def _zeros(labels):
        return torch.zeros_like(labels, dtype=torch.bool)
    return _zeros


class Probe:
    def __init__(self, model, sample):
        self.m = model
        self.vla = model.vla
        self.e, self.t, imgs, state, gt_action = sample
        self.state = np.asarray(state, np.float32)
        self.gt_action = np.asarray(gt_action, np.float32)
        self.m.update_obs(B.make_obs(imgs, state, self.m.instr))
        obs = self.m.observation_window[0]
        self.pv = torch.cat(
            [self.m.processor.image_processor.apply_transform(Image.fromarray(im)) for im in obs["images_raw"]],
            dim=0,
        ).unsqueeze(0).to(torch.bfloat16).to(DEV)
        pn = np.clip(2.0 * (self.state - self.m.pmin) / (self.m.pmax - self.m.pmin + 1e-8) - 1.0, -1, 1)
        self.proprio = torch.from_numpy(pn.astype(np.float32)).unsqueeze(0).to(DEV)
        self.pids = self.m._prompt_ids()
        self.plen = int(self.pids.shape[1])

        # 检查点训练时用的是旧编码（legacy）：token = 31743 + digitize + 1
        self.legacy_idx = AC.encode_indices(
            AC.normalize(self.gt_action, self.m.q01, self.m.q99), mode=AC.MODE_LEGACY)
        self.aid = torch.tensor(
            [[AC.ACTION_TOKEN_BEGIN_IDX + 1 + int(x) for x in self.legacy_idx]],
            dtype=torch.long, device=DEV)
        self.stop = torch.full((1, 1), AC.STOP_INDEX, dtype=torch.long, device=DEV)
        self.inp = torch.cat([self.pids, self.aid, self.stop], dim=1)
        self.lab = torch.cat(
            [torch.full_like(self.pids, PROMPT_IGNORE), self.aid, self.stop], dim=1)
        self.n_vis = None      # 视觉 token 数（插入在位置 0 之后）
        self.captured = {}

    # ---------------------------------------------------------------- helpers
    def _hook(self, tag):
        def hook(module, args, kwargs, output):
            ie = kwargs.get("inputs_embeds", None)
            if ie is not None:
                self.captured[tag] = ie.detach().float().cpu()
        return hook

    def _full_seq(self, mask_on, input_ids=None, labels=None, output_attentions=False):
        ids = self.inp if input_ids is None else input_ids
        lab = self.lab if labels is None else labels
        orig = self.vla._process_action_masks
        if not mask_on:
            self.vla._process_action_masks = masked_off(self.vla)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.vla(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    pixel_values=self.pv,
                    labels=lab,
                    proprio=self.proprio,
                    proprio_projector=self.m.proprio_projector,
                    output_attentions=output_attentions,
                )
        finally:
            self.vla._process_action_masks = orig
        if self.n_vis is None:
            self.n_vis = int(out.logits.shape[1]) - int(ids.shape[1])
        return out

    def action_metrics(self, out):
        """从全序列 logits 里取 14 个动作位置的预测（position p0+k-1 预测第 k 个动作）。"""
        res = []
        for k in range(14):
            pos = self.n_vis + self.plen + k - 1
            res.append(window_probs(out.logits[0, pos]))
        idx = np.array([r["idx"] for r in res], dtype=np.int64)
        act = AC.actions_from_token_ids(AC.token_ids(idx), self.m.q01, self.m.q99, mode=AC.MODE_FIXED)
        return res, idx, np.asarray(act, np.float32)

    # ------------------------------------------------------------------ stages
    def stage_mask_values(self):
        masks = self.vla._process_action_masks(self.lab)
        pos = list(range(self.plen, self.plen + 14))
        vals = [bool(masks[0, p]) for p in pos]
        # 顺带看 prompt 段与 STOP 位置是否被误伤
        prompt_hits = int(masks[0, :self.plen].sum().item())
        stop_hit = bool(masks[0, self.plen + 14])
        return {"action_positions": vals, "n_true": int(sum(vals)),
                "prompt_positions_masked": prompt_hits, "stop_position_masked": stop_hit}

    def stage_embedding_zero(self):
        h = self.vla.language_model.register_forward_hook(self._hook("masked"), with_kwargs=True)
        try:
            self._full_seq(mask_on=True)
        finally:
            h.remove()
        h = self.vla.language_model.register_forward_hook(self._hook("kept"), with_kwargs=True)
        try:
            self._full_seq(mask_on=False)
        finally:
            h.remove()
        out = {}
        for tag in ("masked", "kept"):
            emb = self.captured[tag][0]                       # (S, D)
            p0 = self.n_vis + self.plen
            rows = emb[p0:p0 + 14]
            out[tag] = {
                "max_abs_at_action_positions": float(rows.abs().max()),
                "norm_per_position": [float(x) for x in rows.norm(dim=-1)],
                "prompt_row_norm_example": float(emb[self.n_vis].norm()),
                "vision_row_norm_example": float(emb[1].norm()),
            }
        return out

    def stage_full_seq(self):
        outA = self._full_seq(mask_on=True)
        outB = self._full_seq(mask_on=False)
        rA, iA, aA = self.action_metrics(outA)
        rB, iB, aB = self.action_metrics(outB)
        per_pos = []
        for k in range(14):
            per_pos.append({
                "dim": B.DIM_NAMES[k],
                "idx_masked": int(iA[k]), "idx_kept": int(iB[k]),
                "token_masked": int(AC.token_ids(np.array([iA[k]]))[0]),
                "token_kept": int(AC.token_ids(np.array([iB[k]]))[0]),
                "p_masked": rA[k]["p"], "p_kept": rB[k]["p"],
                "top3_masked": rA[k]["top3"], "top3_kept": rB[k]["top3"],
                "max_prob_diff": float(np.abs(rA[k]["probs"] - rB[k]["probs"]).max()),
                "l1_prob": float(np.abs(rA[k]["probs"] - rB[k]["probs"]).mean()),
                "action_masked": float(aA[k]), "action_kept": float(aB[k]),
                "gt": float(self.gt_action[k]),
            })
        return {
            "tokens_equal": int((iA == iB).sum()),
            "n_positions": 14,
            "action_l1_masked_vs_kept": float(np.abs(aA - aB).mean()),
            "action_max_abs_diff": float(np.abs(aA - aB).max()),
            "A_vs_gt_l1": float(np.abs(aA - self.gt_action).mean()),
            "B_vs_gt_l1": float(np.abs(aB - self.gt_action).mean()),
            "per_position": per_pos,
        }, (aA, aB)

    def stage_free_running(self):
        """逐 token 生成（掩码开/关两种机制），不依赖 model.Model.infer。"""
        def run(mask_on):
            toks, probs = [], []
            for k in range(14):
                if toks:
                    ext = torch.tensor([[AC.ACTION_TOKEN_BEGIN_IDX + 1 + t for t in toks]],
                                       dtype=torch.long, device=DEV)
                    ids = torch.cat([self.pids, ext], dim=1)
                else:
                    ids = self.pids
                lab = ids.clone()
                orig = self.vla._process_action_masks
                if not mask_on:
                    self.vla._process_action_masks = masked_off(self.vla)
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = self.vla(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            pixel_values=self.pv,
                            labels=lab,
                            proprio=self.proprio,
                            proprio_projector=self.m.proprio_projector,
                        )
                finally:
                    self.vla._process_action_masks = orig
                if self.n_vis is None:
                    self.n_vis = int(out.logits.shape[1]) - int(ids.shape[1])
                r = window_probs(out.logits[0, -1])
                toks.append(r["idx"]); probs.append(r["p"])
            idx = np.array(toks, dtype=np.int64)
            act = AC.actions_from_token_ids(AC.token_ids(idx), self.m.q01, self.m.q99, mode=AC.MODE_FIXED)
            return idx, np.asarray(act, np.float32), probs

        iM, aM, pM = run(mask_on=True)
        iK, aK, pK = run(mask_on=False)
        return {"idx_masked_mechanism": iM.tolist(), "idx_kept_mechanism": iK.tolist(),
                "tokens_equal": int((iM == iK).sum()),
                "action_l1": float(np.abs(aM - aK).mean()),
                "action_masked": aM.tolist(), "action_kept": aK.tolist(),
                "top1_prob_masked": pM, "top1_prob_kept": pK}

    def stage_deployment(self):
        """部署路径：KV 缓存逐 token 生成（model.Model.infer）。"""
        act = np.asarray(self.m.infer(self.m.observation_window[0])[0], np.float32)
        dec = self.m.last_decode
        return {"tokens": [int(x) for x in dec["tokens"]],
                "token_ids": [int(x) for x in dec["token_ids"]],
                "top1_prob": [float(p["p"]) for p in dec["probs"]],
                "action": act.tolist()}

    def stage_causality(self):
        seq = torch.cat([self.pids, self.aid], dim=1)          # 无 STOP
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.vla(input_ids=seq, attention_mask=torch.ones_like(seq),
                           output_attentions=True)
        attn = out.attentions[0][0].float()                    # (heads, S, S)
        upper = torch.triu(attn, diagonal=1)
        res = {"n_layers_returned": len(out.attentions),
               "attention_shape": list(attn.shape),
               "max_abs_strictly_upper": float(upper.abs().max()),
               "sum_strictly_upper": float(upper.abs().sum()),
               "is_strictly_causal": bool(float(upper.abs().max()) == 0.0)}

        # 篡改“未来”的动作标记（第 11 个，0-based 10），看过去位置是否被影响
        pos_k = 10
        seq2 = seq.clone()
        seq2[0, self.plen + pos_k] = W0 + 123
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = self.vla(input_ids=seq2, attention_mask=torch.ones_like(seq2))
        diffs = []
        for k in range(14):
            pos = self.plen + k - 1
            d = float((out.logits[0, pos].float() - out2.logits[0, pos].float()).abs().max())
            diffs.append(d)
        res["perturb_future_token_pos"] = pos_k
        res["logit_maxdiff_at_prediction_positions"] = diffs
        res["positions_before_perturb_unchanged"] = all(d == 0.0 for d in diffs[:pos_k])
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="")
    # bootstrap() 会往 sys.argv 里塞 --aloha（prismatic 需要），因此忽略未知参数
    args, _unknown = ap.parse_known_args()

    m = B.load_model()
    sample = B.iter_samples(n=args.sample + 1)[args.sample]
    p = Probe(m, sample)
    print("[probe] sample = %s t=%d ; prompt_len=%d" % (p.e, p.t, p.plen), flush=True)
    print("[probe] legacy idx (checkpoint encoding) = %s" % p.legacy_idx.tolist(), flush=True)
    print("[probe] labels containing PAD(32000): %s" % bool((p.legacy_idx == 256).any()), flush=True)

    res = {"sample": {"episode": p.e, "frame": int(p.t)},
           "legacy_idx": [int(x) for x in p.legacy_idx],
           "gt_action": p.gt_action.tolist(),
           "gt_legacy_has_pad": bool((p.legacy_idx == 256).any())}

    res["mask_values"] = p.stage_mask_values()
    print("\n[1] all_actions_mask at the 14 action positions:")
    print("    %s   (n_true=%d)" % (res["mask_values"]["action_positions"], res["mask_values"]["n_true"]))
    print("    prompt positions masked=%d, STOP position masked=%s" % (
        res["mask_values"]["prompt_positions_masked"], res["mask_values"]["stop_position_masked"]))

    res["embedding_zero"] = p.stage_embedding_zero()
    for tag in ("masked", "kept"):
        d = res["embedding_zero"][tag]
        print("[2] %-6s max|emb| at 14 action positions = %.6f   norms[0:4] = %s" % (
            tag, d["max_abs_at_action_positions"], [round(x, 4) for x in d["norm_per_position"][:4]]))
    print("    reference: vision row norm = %.4f, prompt row norm = %.4f" % (
        res["embedding_zero"]["masked"]["vision_row_norm_example"],
        res["embedding_zero"]["masked"]["prompt_row_norm_example"]))

    full, (aA, aB) = p.stage_full_seq()
    res["full_seq"] = full
    print("\n[3] full-sequence: masked (training) vs kept (autoregressive)")
    print("    %-14s %6s %6s %8s %8s %10s %10s" % ("dim", "idxM", "idxK", "pM", "pK", "actM", "actK"))
    for r in full["per_position"]:
        print("    %-14s %6d %6d %8.4f %8.4f %10.5f %10.5f" % (
            r["dim"], r["idx_masked"], r["idx_kept"], r["p_masked"], r["p_kept"],
            r["action_masked"], r["action_kept"]))
    print("    tokens equal %d/14 ; |action| L1 = %.6f (max %.6f) ; A vs gt L1 = %.6f ; B vs gt L1 = %.6f" % (
        full["tokens_equal"], full["action_l1_masked_vs_kept"], full["action_max_abs_diff"],
        full["A_vs_gt_l1"], full["B_vs_gt_l1"]))

    dep = p.stage_deployment()
    res["deployment"] = dep
    free = p.stage_free_running()
    res["free_running"] = free
    print("\n[4] deployment (KV cache, model.Model.infer) tokens = %s" % dep["tokens"])
    print("    free-running kept-mechanism tokens          = %s (equal %d/14 vs masked-mechanism)" % (
        free["idx_kept_mechanism"], free["tokens_equal"]))
    idx_masked_seq = [r["idx_masked"] for r in full["per_position"]]
    idx_kept_seq = [r["idx_kept"] for r in full["per_position"]]
    print("    deployment tokens vs full-seq(GT-prefix) equal: %d/14 (masked) / %d/14 (kept)" % (
        sum(a == b for a, b in zip(dep["tokens"], idx_masked_seq)),
        sum(a == b for a, b in zip(dep["tokens"], idx_kept_seq))))
    print("    deployment == free-running(kept)?           %s" % (dep["tokens"] == free["idx_kept_mechanism"]))

    res["causality"] = p.stage_causality()
    c = res["causality"]
    print("\n[5] causality: attention shape %s ; max|attn| strictly above diagonal = %.3e ; strictly causal = %s" % (
        c["attention_shape"], c["max_abs_strictly_upper"], c["is_strictly_causal"]))
    print("    perturb future action token #%d -> earlier prediction positions unchanged: %s" % (
        c["perturb_future_token_pos"], c["positions_before_perturb_unchanged"]))
    print("    logit maxdiff per prediction position: %s" % [round(x, 4) for x in c["logit_maxdiff_at_prediction_positions"]])

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("\nwrote " + args.out)
    print("R3_MASK_PROBE_DONE")


if __name__ == "__main__":
    main()
