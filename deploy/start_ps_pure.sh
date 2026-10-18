#!/bin/bash
R=/home/T7/ojh/robot_actual/RoboDojo
OFT=$R/XPolicyLab/policy/OpenVLA_OFT/openvla_oft
export PYTHONPATH=$OFT:$R:$PYTHONPATH
cd $R
exec /home/dgut/pusht_runtime/venv_oft/bin/python XPolicyLab/setup_policy_server.py \
  --config_path XPolicyLab/policy/OpenVLA_Pure/deploy.yml \
  --overrides port=6020 host=localhost bench_name=RoboDojo task_name=push_T \
    policy_name=OpenVLA_Pure action_type=joint action_dim=14 env_cfg_type=arx_x5 seed=0
