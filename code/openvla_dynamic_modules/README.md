# 实际运行的动态加载模块（第一轮审查问题 6 的审查材料）

## 为什么需要这个目录

纯 OpenVLA 部署用 `trust_remote_code=True` 加载基础模型，基础模型 `config.json` 里的
`auto_map` 指向 HF 动态模块：

    "AutoConfig":               "configuration_prismatic.OpenVLAConfig"
    "AutoModelForVision2Seq":   "modeling_prismatic.OpenVLAForActionPrediction"

也就是说，**真正被 import 的 `modeling_prismatic.py` 不是仓库里的同名文件，而是 HF 缓存里
自动生成的一份副本**。第一轮审查时仓库没有收录它，导致“训练/评测/部署到底跑了哪份代码”
无法核对。本目录把实际生效的三份文件原样收录（字节级拷贝，未做任何修改）。

## 实际生效路径

    /home/dgut/.cache/huggingface/modules/transformers_modules/openvla-7b/

策略服务由 `/home/dgut/pusht_runtime/start_ps_pure.sh` 在**宿主机**启动（不在 docker 容器内）：

    cd /home/T7/ojh/robot_actual/RoboDojo
    PYTHONPATH=$OFT:$R
    /home/dgut/pusht_runtime/venv_oft/bin/python XPolicyLab/setup_policy_server.py \
        --config_path XPolicyLab/policy/OpenVLA_Pure/deploy.yml \
        --overrides port=6020 host=localhost bench_name=RoboDojo task_name=push_T \
                    policy_name=OpenVLA_Pure action_type=joint action_dim=14 env_cfg_type=arx_x5 seed=0

因此上面那条 `~/.cache/huggingface/...` 路径就是线上真正执行的代码。

## 文件与 SHA-256（2026-10-18 现场复算）

| 文件 | 字节 | SHA-256 | mtime |
|---|---|---|---|
| modeling_prismatic.py | 48,906 | `b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6` | 2026-09-21 04:17 |
| configuration_prismatic.py | 5,865 | `68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68` | 2026-09-21 04:17 |
| processing_prismatic.py | 12,713 | `2474a5c3fdd4ff15234924636dc0ead6419c89f49699193be47dbb87510a5425` | 2026-09-21 06:12 |

`__init__.py` 是 0 字节占位文件，未收录。

## 与 OFT 源码树的差异

    diff -q .../modeling_prismatic.py       $OFT/prismatic/extern/hf/modeling_prismatic.py       -> 相同
    diff -q .../configuration_prismatic.py  $OFT/prismatic/extern/hf/configuration_prismatic.py  -> 相同
    sha256 亦逐字节一致（b5431a07… / 68cc5ae3…）

也就是说：**实际运行的动态模块与 OFT 源码树里的同名文件完全相同，不存在“仓库版本与运行时
版本不一致”的差异**；第一轮报告里“仓库未收录实际运行文件”的问题由本目录关闭。

## 复现命令

    sha256sum ~/.cache/huggingface/modules/transformers_modules/openvla-7b/*.py
    diff -q ~/.cache/huggingface/modules/transformers_modules/openvla-7b/modeling_prismatic.py \
            /home/dgut/pusht_runtime/openvla_oft_src/prismatic/extern/hf/modeling_prismatic.py

（若在别的机器上复核，路径以 transformers 实际生成的缓存目录为准。）
