
import sys, glob, h5py, numpy as np, cv2, torch
from PIL import Image
R='/home/T7/ojh/robot_actual/RoboDojo'; OFT=R+'/XPolicyLab/policy/OpenVLA_OFT/openvla_oft'
sys.path.insert(0,OFT); sys.path.insert(0,R); sys.argv=['ab','--aloha']
import yaml
cfg=yaml.safe_load(open(R+'/XPolicyLab/policy/OpenVLA_Pure/deploy.yml'))
cfg.update({'task_name':'push_T','env_cfg_type':'arx_x5','action_type':'joint'})
import XPolicyLab.policy.OpenVLA_Pure.model as M
m=M.Model(cfg); print('MODEL_LOADED', flush=True)
D='/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data'
f=h5py.File(sorted(glob.glob(D+'/*.hdf5'))[0],'r'); t=0
def dec(b): return cv2.imdecode(np.frombuffer(bytes(b),np.uint8),cv2.IMREAD_COLOR)[...,::-1].copy()
train_imgs=[dec(f['vision/'+c+'/colors'][t]) for c in ('cam_head','cam_left_wrist','cam_right_wrist')]
train_state=np.concatenate([f['state/left_arm_joint_states'][t],f['state/left_ee_joint_states'][t],
                            f['state/right_arm_joint_states'][t],f['state/right_ee_joint_states'][t]]).astype(np.float32)
dep_imgs=[cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in ('/tmp/f_head.png','/tmp/f_lw.png','/tmp/f_rw.png')]
zero_state=np.zeros(14,dtype=np.float32)
def run(tag, imgs, state):
    obs={"images":{"cam_high":imgs[0],"cam_left_wrist":imgs[1],"cam_right_wrist":imgs[2]},
         "state":state,"instruction":"Push the T-shaped block to align it precisely with the gray T-shaped pad."}
    m.update_obs(obs)
    a=m.infer(m.observation_window[0])[0]
    d=a-state
    print('%s: mean|action-state|=%.5f max=%.5f  action[:8]=%s' % (tag, float(np.abs(d).mean()), float(np.abs(d).max()), np.round(a[:8],4).tolist()), flush=True)
    return a
a1=run('A_trainImg_trainState', train_imgs, train_state)
a2=run('B_depImg_trainState  ', dep_imgs, train_state)
a3=run('C_trainImg_zeroState ', train_imgs, zero_state)
a4=run('D_depImg_zeroState   ', dep_imgs, zero_state)
print('ABLATION_DONE')
