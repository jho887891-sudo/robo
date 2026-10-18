
import json, glob, h5py, numpy as np, os
RT='/home/dgut/pusht_runtime'
D='/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data'
nb=np.load(RT+'/push_T_bounds.npz'); amin=nb['amin']; amax=nb['amax']
def norm(a): return np.clip(2.0*(a-amin)/(amax-amin+1e-8)-1.0,-1,1)
def c14(h,g):
    return np.concatenate([h[g+'/left_arm_joint_states'][:],h[g+'/left_ee_joint_states'][:],
                           h[g+'/right_arm_joint_states'][:],h[g+'/right_ee_joint_states'][:]],axis=1).astype(np.float32)
# fixed samples (held-out 20) if available, else split val
fp=RT+'/fixed_samples.json'
if os.path.exists(fp):
    samples=json.load(open(fp)); print('fixed_samples', len(samples))
else:
    sp=json.load(open(RT+'/split80.json')); val=sp['val'][:20]
    samples=[[e,0] for e in val]; print('split val', len(samples))
L_id=[]; L_prev=[]; gt_norm_all=[]
for e,t in samples:
    with h5py.File(os.path.join(D,e),'r') as h:
        S=c14(h,'state'); A=c14(h,'action')
    t=int(t)
    id_l1=float(np.abs(norm(S[t])-norm(A[t])).mean())
    # constant-hold over the episode (=identity)
    L_id.append(id_l1)
    gt_norm_all.append(norm(A[t]))
print('IDENTITY (action=state) mean L1 over samples: %.4f  median %.4f  max %.4f' % (np.mean(L_id), np.median(L_id), np.max(L_id)))
# also: what L1 would "predict the mean action" give?
mu=np.mean(np.stack(gt_norm_all),0)
print('MEAN-ACTION baseline L1: %.4f' % float(np.mean([np.abs(mu-g).mean() for g in gt_norm_all])))
