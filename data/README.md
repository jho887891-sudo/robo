# 数据与动作表示（**不包含数据集本身**）

## 数据来源
RoboDojo 官方演示数据（ARX-X5，push_T）：本地路径
`/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data/episode_XXXXXXXXXX.hdf5`（100 个 episode）。
数据集与仿真资产**未上传**，可从 RoboDojo-Benchmark 官方渠道获取。

## 单个 episode 的真实结构（实测）
```
action/left_arm_joint_states   (T,6)   action/left_ee_joint_states   (T,1)
action/right_arm_joint_states  (T,6)   action/right_ee_joint_states  (T,1)
state /left_arm_joint_states   (T,6)   state /left_ee_joint_states   (T,1)
state /right_arm_joint_states  (T,6)   state /right_ee_joint_states  (T,1)
vision/cam_head/colors         (T,) bytes(JPEG, 480x640)   + intrinsic_matrix (3,3) / extrinsic_matrix (T,4,4)
vision/cam_left_wrist/...      vision/cam_right_wrist/...
instruction (str)  additional_info/frequency (25)
```

## 动作语义（用数据实测，不靠约定）
在 30 个 episode 上：`mean|A[t] - S[t+1]| = 0.0000`（14 维全 0），`mean|A[t] - S[t]| > 0`
→ **动作 = 下一步的绝对关节目标**；仿真侧由 `robot_manager.set_joint_position_target` 施加（见 `../patches/robodojo_changes.patch` 关联文件）。

## 14 维顺序（训练与部署一致）
`[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]`
- 训练：`c14()` 顺序拼接
- 部署：`XPolicyLab/utils/process_data.py::pack_robot_state(source_type="obs")` 同一顺序
- 两者实测 maxdiff = 0.0

## 归一化统计量（已随仓库提供，非数据）
- `push_T_bounds.npz`（1,206 B）：`amin/amax`（BOUNDS 归一化，用于离线 L1 口径）与 `pmin/pmax`（proprio 归一化）
- `run_meta_*.json`：`q01/q99`（训练集动作 1%/99% 分位，用于离散分箱）
- 复算：`action_stats(train_eps)` 于 `train_pushT_pure2.py`

## 拆分
`split80.json`（2,648 B）：train 80 / val 20 的 episode 文件名列表（随机种子 0 下生成）。
