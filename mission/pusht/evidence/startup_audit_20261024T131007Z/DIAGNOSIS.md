# R28 startup audit — BLOCKED (tool defect at fixed SHA), no code modified

Run: Issue #40 固定命令，工具 SHA `4a1798dfa57551e4a63c977f38927028aea1582f`（`origin/evidence/pusht-startup-data-audit` = `origin/research/pusht-startup-data-audit`）。
执行机 jxxy，`/tmp/pusht-startup-audit-wt-r28a`（detached worktree，HEAD 与要求一致）。

## 结果

```text
exit_code=1
report_exists=false
```

`report.json` 未生成；stdout 为空；stderr 为下列 traceback（原文见本目录 `stderr.log`）：

```text
Traceback (most recent call last):
  File "/tmp/pusht-startup-audit-wt-r28a/tools/pusht_startup_data_audit.py", line 618, in <module>
    sys.exit(main())
  File "/tmp/pusht-startup-audit-wt-r28a/tools/pusht_startup_data_audit.py", line 614, in main
    return run_audit(args)
  File "/tmp/pusht-startup-audit-wt-r28a/tools/pusht_startup_data_audit.py", line 378, in run_audit
    _doc, all_splits, _payload, splits_path, normalizer_path = M2.load_m2_artifacts(
AttributeError: module 'm2_multitask_data' has no attribute 'load_m2_artifacts'
```

## 只读定位（未修改任何代码，未用补丁重跑）

工具内 `M2` / `M2T` 的绑定与调用点：

| 位置 | 代码 | 说明 |
|---|---|---|
| 246 | `import m2_multitask_data as M2` | 第一个导入点 |
| 247 | `import m2_train_shared as M2T` | 与 `M2` 并列 |
| 260 | `M2T.load_m2_artifacts(...)` | 写法正确 |
| 353 | `import m2_multitask_data as M2` | 函数内再次导入，遮蔽同名变量 |
| 378 | `M2.load_m2_artifacts(...)` | **实际抛错点** |

符号实际所在模块（同一 SHA 工作树内实测）：

```text
training/m2_multitask_data.py:234:def load_task_index(task, hdf5_dir=None, verify_targets=False, ...)
training/m2_train_shared.py:821:def load_m2_artifacts(tasks, splits_path=None, normalizer_path=None):

m2_multitask_data  file=/tmp/pusht-startup-audit-wt-r28a/training/m2_multitask_data.py
  has load_m2_artifacts = False
  has load_task_index   = True
m2_train_shared    file=/tmp/pusht-startup-audit-wt-r28a/training/m2_train_shared.py
  has load_m2_artifacts = True
  has load_task_index   = False
```

即：`load_m2_artifacts` 定义在 `m2_train_shared`，不在 `m2_multitask_data`；第 378 行把 `M2`（= m2_multitask_data）当成 `M2T` 使用。
同仓库其它工具（`r16_*.py`、`r24_frame_manifest.py` 等）普遍写成 `import m2_train_shared as M2`，与本工具第 246–247 行的命名约定相反。

## 边界

- 未修改 `tools/pusht_startup_data_audit.py` 或任何仓库文件；未用补丁重跑以『让它变绿』。
- 失败发生在读取 HDF5/RGB **之前**（`run_audit` 第 378 行的契约加载阶段），因此本证据不含任何数据统计结论。
- 未加载模型权重、未启动 CUDA、未写 HDF5/checkpoint/WAL，0 optimizer steps。
- 临时原始目录 `/tmp/pusht-startup-audit-67JI8cz1` 保留在服务器，未清理。

## 仅事实说明（不含建议）

- 该 SHA 的 CPU CI（Actions 40096377406，6 tests）通过，但未覆盖到第 353/378 行的第二次导入路径。
- 失败是模块内符号绑定问题，与数据、HDF5、splits、normalizer、checkpoint 无关：程序在读取这些文件之前就异常退出。

