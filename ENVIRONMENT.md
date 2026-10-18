# 环境与依赖版本

| 组件 | 版本/说明 |
|---|---|
| 主机 | jxxy，32 核 / 58 GB RAM / RTX A6000 48 GB |
| NVIDIA 驱动 | `580.178.04`（CUDA 12.8 运行时；低于此版本 curobo `cuda_core` 会报 `cudaErrorInsufficientDriver`） |
| 仿真容器 | `rd_setup`（`nvcr.io/nvidia/isaac-sim:5.1.0`，`--runtime=nvidia --network host`，Isaac Lab） |
| 策略环境 | `/home/dgut/pusht_runtime/venv_oft`（Python 3.10）：torch 2.4.1+cu121、transformers 4.40.2、tokenizers 0.19.1、peft 0.11.1、timm 0.9.10、numpy 1.26.4、tensorflow-cpu 2.15.1、websockets 16.1.1、msgpack-numpy 0.4.8 |
| 基础模型 | OpenVLA-7B（本地 `openvla_offline_bundle/models/openvla-7b`） |
| 动作常数 | `ACTION_TOKEN_BEGIN_IDX`、`STOP_INDEX`、`IGNORE_INDEX` 来自 `openvla_oft/prismatic/vla/constants.py`（ALOHA/ARX 配置：`NUM_ACTIONS_CHUNK=1`, `ACTION_DIM=14`, `PROPRIO_DIM=14`） |

自检命令：
```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
/home/dgut/pusht_runtime/venv_oft/bin/python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__)"
sudo docker ps --format '{{.Names}}|{{.Status}}' | grep rd_setup
```
