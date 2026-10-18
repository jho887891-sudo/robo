#!/usr/bin/env python3
"""第三轮最小修复补丁：给 modeling_prismatic.py 加动作输入掩码开关，并让自回归脚本显式声明。

只做四件事：
  1. modeling_prismatic.py：新增 set_action_input_masking() / _resolve_action_input_masking()，
     forward 增加 mask_action_inputs 参数；置零动作位置嵌入的那一行改为受开关控制，
     默认仍是 zero（OFT 并行/扩散解码行为完全不变）。
  2. train_pushT_pure2.py：显式 set_action_input_masking("keep")，并在训练/评测前向里传
     mask_action_inputs=False，run_meta 记录该设置。
  3. model.py（部署）：缓存式生成本就不置零（labels=None 分支），这里显式传
     mask_action_inputs=False 并写进 last_decode/动作日志，便于审计“部署用哪种机制”。
  4. 三份 modeling_prismatic.py（仓库副本 / HF 动态模块缓存 / OFT 源码树）保持一致，
     打完补丁后逐字节相同。

幂等：重复运行不会重复插入（用标记检测）。
"""
import hashlib
import io
import os
import shutil
import sys

SUBMISSION = "/home/T7/ojh/robot_actual/submission"
# 注意（第三轮实测）：运行时真正被 import 的动态模块，是 transformers 从
# 基础模型目录里的 .py 复制到 HF 模块缓存之后的那份 —— 也就是说**基础模型目录里的副本才是权威**，
# 只改缓存副本会在下次加载时被覆盖。因此三个位置都要打补丁。
MODELING = [
    SUBMISSION + "/code/openvla_dynamic_modules/modeling_prismatic.py",
    "/home/dgut/pusht_runtime/model_cache/openvla-7b/modeling_prismatic.py",          # 权威来源
    "/home/dgut/.cache/huggingface/modules/transformers_modules/openvla-7b/modeling_prismatic.py",
    "/home/dgut/pusht_runtime/openvla_oft_src/prismatic/extern/hf/modeling_prismatic.py",
]
TRAIN = [SUBMISSION + "/training/train_pushT_pure2.py",
         "/home/dgut/pusht_runtime/train_pushT_pure2.py"]
DEPLOY = [SUBMISSION + "/code/xpolicylab/policy/OpenVLA_Pure/model.py",
          "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_Pure/model.py"]

HELPERS = '''
    # === [R3] Action-Input Masking Control ===
    def set_action_input_masking(self, mode="zero"):
        """选择动作 token 位置输入嵌入的处理方式。

        "zero" : 把动作位置的输入嵌入置零（OFT 并行/扩散解码的原有机制；默认，向后兼容）
        "keep" : 保留前序真实动作标记的嵌入（纯 OpenVLA 自回归训练/离线评测必须用这个）

        修复前该行为是硬编码的：只要传了 labels 就把动作位置置零，于是自回归训练无法
        条件于前序动作标记，而部署侧缓存式生成（labels=None）喂的却是真实标记。
        """
        if mode not in ("zero", "keep"):
            raise ValueError("action_input_masking must be 'zero' or 'keep', got %r" % (mode,))
        self.action_input_masking = mode
        return self.action_input_masking

    def _resolve_action_input_masking(self, flag=None):
        """forward 的 mask_action_inputs 优先；否则回落到实例属性；默认 zero（旧行为）。"""
        if flag is not None:
            return bool(flag)
        return getattr(self, "action_input_masking", "zero") != "keep"
'''

ANCHOR = '''    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask
'''

OLD_SIG = '''        diffusion_timestep_embeddings=None,
        use_film: bool = False,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:'''
NEW_SIG = '''        diffusion_timestep_embeddings=None,
        mask_action_inputs: Optional[bool] = None,
        use_film: bool = False,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:'''

OLD_ZERO = '''            else:
                # Replace the embeddings of the action tokens with zeros
                # (Later on, the positional embeddings will be added to them)
                all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
                input_embeddings = input_embeddings * ~all_actions_mask
'''
NEW_ZERO = '''            else:
                # [R3] 动作位置的输入嵌入处理方式由 mask_action_inputs / self.action_input_masking 决定：
                #   zero（默认，OFT 并行/扩散解码）：置零 => 各动作位置互不可见
                #   keep（纯 OpenVLA 自回归）：保留前序真实动作标记嵌入 => 按因果注意力条件于前序动作
                if self._resolve_action_input_masking(mask_action_inputs):
                    # Replace the embeddings of the action tokens with zeros
                    # (Later on, the positional embeddings will be added to them)
                    all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
                    input_embeddings = input_embeddings * ~all_actions_mask
'''


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def patch_modeling(path):
    s = io.open(path, encoding="utf-8").read()
    if "[R3] Action-Input Masking Control" in s:
        return "already-patched"
    for blob in (ANCHOR, OLD_SIG, OLD_ZERO):
        if blob not in s:
            raise SystemExit("anchor not found in %s:\n%s" % (path, blob[:120]))
    s = s.replace(ANCHOR, ANCHOR + HELPERS, 1)
    s = s.replace(OLD_SIG, NEW_SIG, 1)
    s = s.replace(OLD_ZERO, NEW_ZERO, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    return "patched"


def patch_train(path):
    s = io.open(path, encoding="utf-8").read()
    if "[R3]" in s:
        return "already-patched"
    old = '''    vla.train()
    if pp is not None: pp.train()
    t0 = time.time(); step = 0'''
    new = '''    # [R3] 自回归训练必须保留前序动作标记的正常输入嵌入（修复前被 all_actions_mask 置零）
    if hasattr(vla, "set_action_input_masking"):
        vla.set_action_input_masking("keep")
        log("action_input_masking=keep (autoregressive: previous action tokens kept)")
    else:
        log("WARNING: model has no set_action_input_masking(); action inputs may be zeroed")
    vla.train()
    if pp is not None: pp.train()
    t0 = time.time(); step = 0'''
    if old not in s:
        raise SystemExit("train anchor1 missing")
    s = s.replace(old, new, 1)

    # 训练前向
    old_fwd = '''            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv, labels=lab,
                      proprio=pr, proprio_projector=pp)'''
    new_fwd = '''            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv, labels=lab,
                      proprio=pr, proprio_projector=pp, mask_action_inputs=False)  # [R3] 保留前序动作嵌入'''
    if old_fwd not in s:
        raise SystemExit("train anchor2 missing")
    s = s.replace(old_fwd, new_fwd, 1)

    # 评测前向（完整序列）
    old_ev = '''                out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                          labels=inp.clone(), proprio=pr, proprio_projector=pp)'''
    new_ev = '''                out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                          labels=inp.clone(), proprio=pr, proprio_projector=pp,
                          mask_action_inputs=False)  # [R3] 与部署机制一致'''
    if old_ev not in s:
        raise SystemExit("train anchor3 missing")
    s = s.replace(old_ev, new_ev, 1)

    old_meta = '''               "codec_mode": CODEC_MODE,'''
    new_meta = '''               "codec_mode": CODEC_MODE,
               "action_input_masking": "keep",  # [R3] 自回归：保留前序动作标记嵌入'''
    if old_meta not in s:
        raise SystemExit("train anchor4 missing")
    s = s.replace(old_meta, new_meta, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    return "patched"


def patch_deploy(path):
    s = io.open(path, encoding="utf-8").read()
    if "[R3]" in s:
        return "already-patched"
    # 1) 前缀前向：显式声明不置零（提示段本来没有动作 token，这里只是把机制写清楚）
    old_p = '''                out = self.vla(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    pixel_values=pixel_values,
                    labels=prompt_ids.clone(),
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                    use_cache=True,
                )'''
    new_p = '''                out = self.vla(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    pixel_values=pixel_values,
                    labels=prompt_ids.clone(),
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                    use_cache=True,
                    mask_action_inputs=False,   # [R3] 前缀不置零动作位置（与训练/评测机制一致）
                )'''
    if old_p not in s:
        raise SystemExit("deploy anchor1 missing")
    s = s.replace(old_p, new_p, 1)
    # 2) 非缓存的全量重算路径
    old_n = '''                    out = self.vla(
                        input_ids=model_input,
                        attention_mask=torch.ones_like(model_input),
                        pixel_values=pixel_values,
                        labels=model_input.clone(),
                        proprio=proprio,
                        proprio_projector=self.proprio_projector,
                    )'''
    new_n = '''                    out = self.vla(
                        input_ids=model_input,
                        attention_mask=torch.ones_like(model_input),
                        pixel_values=pixel_values,
                        labels=model_input.clone(),
                        proprio=proprio,
                        proprio_projector=self.proprio_projector,
                        mask_action_inputs=False,   # [R3] 自回归：保留前序动作标记嵌入
                    )'''
    if old_n not in s:
        raise SystemExit("deploy anchor2 missing")
    s = s.replace(old_n, new_n, 1)
    # 3) 记录机制
    old_rec = '''        self.last_decode = {
            "tokens": [int(t) for t in tokens],'''
    new_rec = '''        self.last_decode = {
            "action_input_masking": "keep",   # [R3] 部署侧逐 token 生成：不置零
            "tokens": [int(t) for t in tokens],'''
    if old_rec not in s:
        raise SystemExit("deploy anchor3 missing")
    s = s.replace(old_rec, new_rec, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    return "patched"


def main():
    log = []
    for p in MODELING:
        if not os.path.exists(p):
            log.append("MISSING %s" % p)
            continue
        if not os.path.exists(p + ".r3bak") and os.path.exists(p):
            shutil.copy2(p, p + ".r3bak")
        log.append("%s %s" % (patch_modeling(p), p))
    for p in TRAIN:
        log.append("%s %s" % (patch_train(p), p))
    for p in DEPLOY:
        log.append("%s %s" % (patch_deploy(p), p))
    print("\n".join(log))
    print("\n--- hashes (modeling copies must be identical) ---")
    hs = {}
    for p in MODELING:
        if os.path.exists(p):
            hs[p] = sha(p)
            print("%s  %s" % (hs[p], p))
    print("all_modeling_identical=%s" % (len(set(hs.values())) == 1))
    print("R3_PATCH_DONE")


if __name__ == "__main__":
    main()
