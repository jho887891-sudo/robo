#!/usr/bin/env python3
"""统一动作离散化编解码（256-bin 动作 token）——训练 / 离线评测 / 部署共用的唯一实现。

为什么需要它
------------
修复前的实现分散在训练脚本 train_pushT_pure2.py（编码）、部署 model.py（解码）和
三个诊断脚本里，各写一遍 np.digitize / BIN_CENTERS，导致下面两个真实缺陷：

  (D1) 编码多加了 1：np.digitize(v, LEVELS) 的取值是 1..256，却又写成
       token = ACTION_TOKEN_BEGIN_IDX + idx + 1，于是训练标签落在 31745..32000，
       而部署解码窗口是 31744..31999 —— 训练目标整体比解码窗口偏移 1 个 token。
  (D2) 边界溢出：v >= q99 时归一化被 clip 到 +1.0，np.digitize(+1.0) 返回 256，
       token = 31743 + 256 + 1 = 32000，正好等于 pad_token_id（PAD），
       既不在合法动作 token 窗口内，也永远不可能被部署端的 argmax 选出。
       实测 push_T 标签中 12.23% 是 32000（97.70% 的样本至少含 1 个）。

修复后（MODE_FIXED）的约定
--------------------------
    ACTION_TOKEN_BEGIN_IDX = 31743
    合法动作 token         = BEGIN+1 .. BEGIN+256 = 31744 .. 31999
    PAD_TOKEN_ID           = 32000（text_config.pad_token_id，禁止作为动作目标）
    STOP_INDEX             = 2

    idx   = clip(digitize(clip(v,-1,1), LEVELS) - 1, 0, 255)   # 0..255，满射
    token = ACTION_TOKEN_BEGIN_IDX + idx + 1                   # 31744..31999
    v_hat = LEVELS[idx]                                        # LEVELS = linspace(-1,1,256)

    不变量：
      * 任意 v in [-1, 1] 都映射到合法动作 token，永不产生 PAD / STOP / 越界 id；
      * idx=0 对应 v=-1 精确，idx=255 对应 v=+1 精确（端点可表示）；
      * 量化误差 |v_hat - v| <= 2/255 = 0.00784（[-1,1] 单位）；
      * 256 个 idx 全部可达（对 v 均匀采样时每个 idx 都被用到）。

MODE_LEGACY 只用于复刻已训练好的 6000 步检查点（它的标签是用旧编码烧进去的），
不得用于任何新训练。MODE_LEGACY 下的两个缺陷由 tests/test_action_codec.py 断言记录。

本模块只依赖 numpy，import 时不触碰 torch，因此单元测试可在纯 CPU 环境跑。
"""
import numpy as np

# ---------------------------------------------------------------- 常量（OpenVLA / OFT）
ACTION_TOKEN_BEGIN_IDX = 31743
N_BINS = 256
LEGAL_TOKEN_MIN = ACTION_TOKEN_BEGIN_IDX + 1        # 31744
LEGAL_TOKEN_MAX = ACTION_TOKEN_BEGIN_IDX + N_BINS   # 31999
PAD_TOKEN_ID = 32000
STOP_INDEX = 2

# 重建格点：沿用代码库既有的 linspace(-1, 1, 256)，使修复只是“去掉多余偏移”，
# 不改变已训练权重的取值栅格。
LEVELS = np.linspace(-1.0, 1.0, N_BINS).astype(np.float32)
MAX_QUANT_ERROR = float(2.0 / (N_BINS - 1))         # 0.0078431...

MODE_FIXED = "fixed"
MODE_LEGACY = "legacy"
MODES = (MODE_FIXED, MODE_LEGACY)


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError("unknown codec mode %r (expected one of %s)" % (mode, MODES))
    return mode


# ------------------------------------------------------------------- 归一化 / 反归一化
def normalize(actions, q01, q99):
    """物理动作 -> [-1, 1]（越界 clip）。q01/q99 为训练集动作的 1%/99% 分位。"""
    a = np.asarray(actions, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    return np.clip(2.0 * (a - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)


def denormalize(norm, q01, q99):
    """[-1, 1] -> 物理动作。"""
    n = np.asarray(norm, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    return ((n + 1.0) / 2.0 * (q99 - q01 + 1e-8) + q01).astype(np.float32)


# ------------------------------------------------------------------------- 编码 / 解码
def encode_indices(norm, mode=MODE_FIXED):
    """归一化动作 -> bin 索引。

    MODE_FIXED  : 0..255（合法、满射，与解码窗口一一对应）
    MODE_LEGACY : 1..256（复刻旧检查点的训练标签；256 会变成 PAD，见 D2）
    """
    _check_mode(mode)
    v = np.clip(np.asarray(norm, dtype=np.float32), -1.0, 1.0)
    d = np.digitize(v, LEVELS)
    if mode == MODE_FIXED:
        return np.clip(d - 1, 0, N_BINS - 1).astype(np.int64)
    return np.asarray(d, dtype=np.int64)


def decode_indices(idx, mode=MODE_FIXED):
    """bin 索引 -> 归一化动作（LEVELS 格点）。

    两种 mode 的查表方式相同（LEVELS[idx]），差别只在索引的合法范围：
    固定编码的 idx 就是解码窗口下标；旧编码的 idx 比窗口下标大 1。
    """
    _check_mode(mode)
    i = np.clip(np.asarray(idx, dtype=np.int64), 0, N_BINS - 1)
    return LEVELS[i].astype(np.float32)


def token_ids(idx):
    """bin 索引（0..255）-> 动作 token id（31744..31999）。"""
    i = np.asarray(idx, dtype=np.int64)
    return (ACTION_TOKEN_BEGIN_IDX + 1 + i).astype(np.int64)


def indices_from_token_ids(ids):
    """动作 token id -> bin 索引（0..255），越界/非动作 token 抛异常。"""
    i = np.asarray(ids, dtype=np.int64)
    validate_token_ids(i)
    return (i - (ACTION_TOKEN_BEGIN_IDX + 1)).astype(np.int64)


def decode_token_ids(ids, mode=MODE_FIXED):
    """动作 token id -> 归一化动作（带合法性校验）。"""
    return decode_indices(indices_from_token_ids(ids), mode=mode)


# --------------------------------------------------------------------------- 合法性校验
def validate_token_ids(ids):
    """校验所有 id 都在合法动作 token 区间内，返回 (min, max, n)。

    非法情形（PAD / STOP / 越界）一律抛 ValueError，绝不静默放过。
    """
    i = np.asarray(ids, dtype=np.int64)
    if i.size == 0:
        return (None, None, 0)
    bad = (i < LEGAL_TOKEN_MIN) | (i > LEGAL_TOKEN_MAX)
    if bad.any():
        u = np.unique(i[bad])[:8].tolist()
        raise ValueError(
            "illegal action token id(s) %s; legal window is [%d, %d] "
            "(PAD=%d, STOP=%d)" % (u, LEGAL_TOKEN_MIN, LEGAL_TOKEN_MAX, PAD_TOKEN_ID, STOP_INDEX)
        )
    return (int(i.min()), int(i.max()), int(i.size))


def assert_legal_token_ids(ids):
    """同 validate_token_ids，但只校验不返回（训练循环里直接调用）。"""
    validate_token_ids(ids)
    return True


def token_ids_for_actions(actions, q01, q99, mode=MODE_FIXED):
    """物理动作 -> 动作 token id（编码全链路，训练脚本用这个入口）。"""
    _check_mode(mode)
    idx = encode_indices(normalize(actions, q01, q99), mode=mode)
    ids = token_ids(idx) if mode == MODE_FIXED else (ACTION_TOKEN_BEGIN_IDX + 1 + idx)
    validate_token_ids(ids)
    return np.asarray(ids, dtype=np.int64)


def actions_from_token_ids(ids, q01, q99, mode=MODE_FIXED):
    """动作 token id -> 物理动作（解码全链路，部署/评测用这个入口）。"""
    return denormalize(decode_token_ids(ids, mode=mode), q01, q99)


# ------------------------------------------------------------------------------- 自检
def roundtrip(norm_values):
    """返回 (idx, decoded, abs_err) —— 供测试与报告使用。"""
    v = np.asarray(norm_values, dtype=np.float32)
    idx = encode_indices(v, MODE_FIXED)
    dec = decode_indices(idx, MODE_FIXED)
    return idx, dec, np.abs(dec - v)


def summary():
    return {
        "n_bins": N_BINS,
        "legal_token_window": [LEGAL_TOKEN_MIN, LEGAL_TOKEN_MAX],
        "pad_token_id": PAD_TOKEN_ID,
        "levels": [float(LEVELS[0]), float(LEVELS[-1])],
        "max_quant_error": MAX_QUANT_ERROR,
        "modes": list(MODES),
        "default_mode": MODE_FIXED,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=1))
