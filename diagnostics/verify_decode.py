
import sys, glob, json, h5py, numpy as np, cv2, torch
from PIL import Image
R='/home/T7/ojh/robot_actual/RoboDojo'; OFT=R+'/XPolicyLab/policy/OpenVLA_OFT/openvla_oft'
sys.path.insert(0,OFT); sys.path.insert(0,R); sys.argv=['v','--aloha']
import yaml
cfg=yaml.safe_load(open(R+'/XPolicyLab/policy/OpenVLA_Pure/deploy.yml'))
cfg.update({'task_name':'push_T','env_cfg_type':'arx_x5','action_type':'joint'})
import XPolicyLab.policy.OpenVLA_Pure.model as M
nb=np.load(cfg['bounds_path']); print('PRIO pmin', np.round(nb['pmin'],4).tolist()); print('PRIO pmax', np.round(nb['pmax'],4).tolist())
model=M.Model(cfg); print('MODEL_LOADED', flush=True)
D='/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data'
f=h5py.File(sorted(glob.glob(D+'/*.hdf5'))[0],'r'); t=50
def dec(b): return cv2.imdecode(np.frombuffer(bytes(b),np.uint8),cv2.IMREAD_COLOR)[...,::-1].copy()
st=np.concatenate([f['state/left_arm_joint_states'][t],f['state/left_ee_joint_states'][t],
                   f['state/right_arm_joint_states'][t],f['state/right_ee_joint_states'][t]]).astype(np.float32)
obs={"vision":{c:{"color":dec(f['vision/'+c+'/colors'][t])} for c in ('cam_head','cam_left_wrist','cam_right_wrist')},
     "state":{"left_arm_joint_state":st[0:6],"left_ee_joint_state":st[6:7],
              "right_arm_joint_state":st[7:13],"right_ee_joint_state":st[13:14]},
     "instruction":"Push the T-shaped block to align it precisely with the gray T-shaped pad."}
model.update_obs(obs)
enc=model.observation_window[0]
# capture tokens by monkeypatching torch.argmax? simpler: replicate decode here
import torch as T
pv=T.cat([model.processor.image_processor.apply_transform(Image.fromarray(x)) for x in enc['images_raw']],dim=0).unsqueeze(0).to(T.bfloat16).to(model.device)
pr_np=np.clip(2.0*(enc['state']-model.pmin)/(model.pmax-model.pmin+1e-8)-1.0,-1,1).astype(np.float32)
pr=T.from_numpy(pr_np).unsqueeze(0).to(model.device)
pids=model._prompt_ids()
toks=[]
with T.no_grad(), T.autocast('cuda',dtype=T.bfloat16):
    out=model.vla(input_ids=pids, attention_mask=T.ones_like(pids), pixel_values=pv, labels=pids.clone(),
                  proprio=pr, proprio_projector=model.proprio_projector, use_cache=True)
    past=out.past_key_values; lg=out.logits[0,-1]
    for k in range(14):
        B=M.ACTION_TOKEN_BEGIN_IDX
        cand=lg[B+1:B+1+256]; tok=int(T.argmax(cand).item()); toks.append(tok)
        if k==13: break
        nxt=T.tensor([[B+tok+1]],dtype=pids.dtype,device=model.device)
        out=model.vla(input_ids=nxt, past_key_values=past, labels=None, use_cache=True)
        past=out.past_key_values; lg=out.logits[0,-1]
print('TOKENS', toks)
print('TOKEN_RANGE', min(toks), max(toks), 'in_window', all(0<=x<=255 for x in toks))
C=np.linspace(-1,1,256).astype(np.float32)
act=(C[np.clip(np.array(toks),0,255)]+1.0)/2.0*(model.q99-model.q01+1e-8)+model.q01
gt=np.concatenate([f['action/left_arm_joint_states'][t],f['action/left_ee_joint_states'][t],
                   f['action/right_arm_joint_states'][t],f['action/right_ee_joint_states'][t]]).astype(np.float32)
print('PRED', np.round(act,4).tolist())
print('GT  ', np.round(gt,4).tolist())
print('ABS_ERR', np.round(np.abs(act-gt),4).tolist())
print('L1', float(np.abs(act-gt).mean()))
print('PROPRIO_NORM', np.round(pr_np,3).tolist())
