
import sys, glob, h5py, numpy as np, cv2
R='/home/T7/ojh/robot_actual/RoboDojo'; OFT=R+'/XPolicyLab/policy/OpenVLA_OFT/openvla_oft'
sys.path.insert(0,OFT); sys.path.insert(0,R); sys.argv=['v','--aloha']
import yaml
cfg=yaml.safe_load(open(R+'/XPolicyLab/policy/OpenVLA_Pure/deploy.yml'))
cfg.update({'task_name':'push_T','env_cfg_type':'arx_x5','action_type':'joint'})
import XPolicyLab.policy.OpenVLA_Pure.model as M
m=M.Model(cfg); print('MODEL_LOADED', flush=True)
D='/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data'
eps=sorted(glob.glob(D+'/*.hdf5'))
def dec(b): return cv2.imdecode(np.frombuffer(bytes(b),np.uint8),cv2.IMREAD_COLOR)[...,::-1].copy()
acts=[]; gts=[]
for k,(ep,t) in enumerate([(eps[0],0),(eps[1],0),(eps[2],100),(eps[3],200),(eps[4],50),(eps[5],300),(eps[6],10),(eps[7],150)]):
    with h5py.File(ep,'r') as f:
        n=int(f['action/left_arm_joint_states'].shape[0]); t=min(int(t), n-1)
        imgs=[dec(f['vision/'+c+'/colors'][t]) for c in ('cam_head','cam_left_wrist','cam_right_wrist')]
        st=np.concatenate([f['state/left_arm_joint_states'][t],f['state/left_ee_joint_states'][t],
                           f['state/right_arm_joint_states'][t],f['state/right_ee_joint_states'][t]]).astype(np.float32)
        gt=np.concatenate([f['action/left_arm_joint_states'][t],f['action/left_ee_joint_states'][t],
                           f['action/right_arm_joint_states'][t],f['action/right_ee_joint_states'][t]]).astype(np.float32)
    obs={"images":{"cam_high":imgs[0],"cam_left_wrist":imgs[1],"cam_right_wrist":imgs[2]},
         "state":st,"instruction":"Push the T-shaped block to align it precisely with the gray T-shaped pad."}
    m.update_obs(obs); a=m.infer(m.observation_window[0])[0]
    acts.append(a); gts.append(gt)
A=np.stack(acts); G=np.stack(gts)
np.set_printoptions(precision=4, suppress=True, linewidth=220)
print('n_obs', len(A))
print('PRED std across obs :', A.std(0))
print('GT   std across obs :', G.std(0))
print('PRED mean           :', A.mean(0))
print('identity L1 (pred vs own state) not shown; L1 vs GT:', float(np.abs(A-G).mean()))
print('L1 of mean-pred vs GT:', float(np.abs(A.mean(0)-G).mean()))
print('pairwise max |pred_i - pred_j|:', float(np.abs(A[:,None,:]-A[None,:,:]).max()))
print('flag: identical predictions across all obs?', bool(np.allclose(A, A[0:1], atol=1e-6)))
