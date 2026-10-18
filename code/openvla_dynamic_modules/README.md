# 实际运行的动态加载模块（第一轮问题 6 / 第三轮问题 6 的审查材料）

## 加载链（第三轮修正后的准确描述）

基础模型 `config.json` 的 `auto_map`：

    "AutoConfig":             "configuration_prismatic.OpenVLAConfig"
    "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"

真实加载链是**两级复制**，第三轮实测确认：

    /home/dgut/pusht_runtime/model_cache/openvla-7b/modeling_prismatic.py      ← 权威来源
            │ transformers 加载时复制
            ▼
    /home/dgut/.cache/huggingface/modules/transformers_modules/openvla-7b/     ← 实际被 import
            modeling_prismatic.py

第二轮只记录了缓存路径，没记录“缓存是从基础模型目录复制出来的”这一点，导致第三轮第一次打补丁
只改了缓存副本、被下次加载覆盖（实测报 `AttributeError ... no such attribute '_resolve_action_input_masking'`）。
**结论：要改运行代码，必须改基础模型目录里的那份。**

## 文件与 SHA-256

| 文件 | 字节 | SHA-256 | 说明 |
|---|---|---|---|
| modeling_prismatic.py（本轮修复后） | 50,580 | `8d89d5f09d483651dee57ba2d672c2a901396e9a7ea5bb3e2e66e2d743eeefff` | 含第三轮 `mask_action_inputs` 开关，默认 `zero` |
| modeling_prismatic.py（修复前） | 48,906 | `b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6` | 动作输入无条件置零 |
| configuration_prismatic.py | 5,865 | `68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68` | 未改动 |
| processing_prismatic.py | 12,713 | `2474a5c3fdd4ff15234924636dc0ead6419c89f49699193be47dbb87510a5425` | 未改动 |

本目录里的 `modeling_prismatic.py` 是本轮修复后的版本；修复前的原件保存在
`modeling_prismatic.py.r3bak`（由补丁脚本生成，与本文件同目录，若存在）。

## 与 OFT 源码树的差异

修复前：与 `openvla_oft_src/prismatic/extern/hf/modeling_prismatic.py` **字节一致**。
修复后：两处都打了同一个补丁，仍**字节一致**（同 `8d89d5f0…`），因此“仓库副本 = 运行时 = OFT 源码树”。

## 复现命令

    sha256sum /home/dgut/pusht_runtime/model_cache/openvla-7b/modeling_prismatic.py \
              /home/dgut/.cache/huggingface/modules/transformers_modules/openvla-7b/modeling_prismatic.py \
              /home/dgut/pusht_runtime/openvla_oft_src/prismatic/extern/hf/modeling_prismatic.py
    # 三行应完全相同
