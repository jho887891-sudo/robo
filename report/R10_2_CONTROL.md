# R10.2 D5 — 控制时序与离线指标工具箱（CPU，latency = NOT_MEASURED）

**状态**：`IMPLEMENTED + VERIFIED_CPU`（`tools/r10_2_control_timing.py`、`tools/r10_2_offline_eval.py`）

---

## 1. 控制时间轴

* 常量：`FRAME_MS = 40.0`（25 Hz，来自真实 demo 数据），记录但**不假设**旧部署 ~0.93 s 的动作保持口径；
* `stage_timestamps()`：capture → queue → prefill → decode14 → sim_step → hold 六段逐段起止时间；
* `budget_check()`：与 40 ms 帧预算比较并报超时量；**超帧不等于不安全**（动作保持），只是不能声称实时 25 Hz；
* `summarize()`：有真实样本才给 p50/p95/p99，否则 `status=NOT_MEASURED`（本轮就是 NOT_MEASURED）。

## 2. 离线指标（与 R7–R10 同口径）

physical 14D L1（**先帧内 14 维平均，再回合宏平均**）、per-dim/per-arm、cosine、sign（带 `|Δ|≥1e-3`、`|R|≥1e-4` 门槛）、
运动强度分层、clip rate、修正量 R² = 1 − Var(R̂−R)/Var(R)。

回归用**手算真值**锁死口径：例如 4 帧中仅一个 0.001 偏差 → L1 = 0.001/(4×14)（早前把 14 维平均漏掉，测试当场报错）。

## 3. 未做 / 未声称

* 没有真实 7B 闭环运行，因此**没有任何延迟数字**，也没有闭环成功率；
* 官方 success/score、Wilson 置信区间报告器只搭了读取历史 JSON 的骨架，未跑新闭环。