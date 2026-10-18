#!/usr/bin/env python3
import os, json, random, h5py
RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
HDF5_DIR = RT + "/hdf5_push_T/arx_x5/data"
SPLIT = RT + "/run80_cont/split.json"
N = 20
val = json.load(open(SPLIT))["val"]
random.seed(11)
out = []
for _ in range(N):
    e = random.choice(val)
    with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
        T = h["action/left_arm_joint_states"].shape[0]
    out.append([e, random.randrange(0, max(1, T - 25 - 1))])
json.dump(out, open(RT + "/fixed_samples.json", "w"), indent=1)
print("FIXED_SAMPLES", len(out))
print(out[:3])
