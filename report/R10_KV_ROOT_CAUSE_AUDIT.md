# R10 KV 根因诊断 — 前置条件已就位，GPU 窗口 BLOCKED_RESOURCE

**状态**：`BLOCKED_RESOURCE`（授权已记录，诊断**未执行**）｜ **分支** `r10/cpu-gates`
**结论**：不声称任何根因。R8 的 8/14 失配仍是 `OPEN_UNRESOLVED`。

---

## 1. 为什么必须先做 CPU 补丁（已完成）

R8/审计版本的 `kv_diagnose` 只保存 `max/mean Δlogit`、margin、argmax —— **没有完整 256-bin logits**，
因此即使跑完也无法事后复算"数值 vs 结构"。R10 要求先补 sidecar，再上 GPU。

| 补丁 | 内容 | 证据 |
|---|---|---|
| logits sidecar | 每步 `logits_kv[14,256]`、`logits_full[14,256]`、token 列、step、column_index 存入 `r10_kv_logits.npz`；JSON 记 sha256/bytes/shape/dtype/列定义 | `evidence/r10/probe_patch.json`、`test_r10_kv_probe.log`（16/16） |
| 缓存 preflight | 显式 `gradient_checkpointing_disable()`、强制 `config.use_cache=True`、`cached is None` 即抛错拒绝；记录 before/after 状态 | 同上（3 项 + 静态 4 项） |
| 每步语义 | 记录 `prefix_len_full`、`cache_len_kv`、`step0_prefix_only`；继续用 full-path greedy token teacher-force | 同上 |
| 稳态常量 | `W0=31744`、`BINS=256` 未变；模型数学未改 | `kv_window_constants_unchanged` |
| 分类 | 记录 `mismatch_class`（`numeric_plausible` / `structural_suspect`，阈值 0.05 仅作启发式）与 `KV_EQUIVALENT` 独立标志 | 源码 `DIAG_EXECUTED`/`KV_EQUIVALENT` 标记 |

## 2. 资源实测（决定 BLOCKED）

```
total 47.402 GiB | free 23.300 GiB | occupied by others 24.102 GiB | required >= 24.000 GiB
pid 3111650  /isaac-sim/kit/kit                       2486 MiB
pid 3451404  /isaac-sim/kit/python/bin/python3        5604 MiB
pid 3469786  /home/dgut/pusht_runtime/venv_oft/bin/python 16200 MiB
container    rd_setup  nvcr.io/nvidia/isaac-sim:5.1.0  (running push_T eval, 10 envs, policy server :6020)
```

缺口 **0.700 GiB**。按 R10 §3「没有空闲或资源冲突则 `BLOCKED_RESOURCE`」，本轮**不启动**诊断，
不终止、不抢占、不重试第二次。原始采样在 `evidence/r10/gpu_resource_probe.json`。

## 3. 一旦窗口具备，将执行的唯一命令（原样，不重设计）

```bash
task=/home/T7/ojh/robot_actual/r10/kv; mkdir -p "$task"
cd /home/T7/ojh/robot_actual/r10/cpu
timeout 300s python diagnostics/r8_smoke.py --kv-diagnose \
  --norm-contract evidence/r9_gate_20261020/frozen_contract_64train.json \
  --min-free-gb 24 --out "$task/$(date -u +%Y%m%dT%H%M%SZ)" 2>&1 | tee "$task/run.log"
```

准入前提（本轮已全部满足，可复算）：CPU P0 全过；使用 64 TRAIN-only 冻结契约（sha `2ab68fd3…`）；
`S,V,Acc` signed-motion q99；无训练、无 optimizer。

产出：`r8_kv_diagnose.json` + `r10_kv_logits.npz`（sha256 可独立校验）+ 每步 margin/Δlogit/长度。

## 4. 判定规则（写死，事中不改）

| 情况 | 状态标记 | 含义 |
|---|---|---|
| 14/14 一致 | `DIAG_EXECUTED + KV_EQUIVALENT` | 缓存路径等价，Gate C 功能通过 |
| 失配且 max\|Δlogit\| ≤ 0.05 | `DIAG_EXECUTED + OPEN_UNRESOLVED`（numeric_plausible） | 翻转与数值差兼容，**不是** bf16 证据 |
| 失配且 max\|Δlogit\| > 0.05 | `DIAG_EXECUTED + OPEN_UNRESOLVED`（structural_suspect） | 优先查 cache/position/mask/token，禁止二次 GPU |
| 超时/OOM/cache 为 None/资源冲突 | `BLOCKED_RESOURCE` + 原始 stderr | 立即停止，不重试 |

## 5. 当前不能声称的

不是 bf16 根因、不是 mask 错误、不是"诊断已做"。R8 的 8/14 记录与旧 logits 缺失的事实一并保留。
