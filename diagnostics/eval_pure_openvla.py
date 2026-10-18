#!/usr/bin/env python3
"""Corrected evaluation for the PURE OpenVLA push_T checkpoint.

Two metrics on the 20 held-out episodes:
  (1) teacher-forced token accuracy (GT prefix fed in)
  (2) AUTOREGRESSIVE decoded action L1, normalized with the SAME OFT BOUNDS stats
      so it is directly comparable with the OFT run (step200 = 0.2062, cont300 = 0.1667)
"""
import os, sys, json, random
import numpy as np, torch, h5py, cv2
from PIL import Image

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
MODEL = os.environ.get("PUSHT_MODEL", RT + "/models_openvla-7b")
HDF5_DIR = os.environ.get("PUSHT_HDF5", RT + "/hdf5_push_T/arx_x5/data")
OFT = os.environ.get("PUSHT_OFT", RT + "/openvla_oft_src")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
CKPTS = os.environ.get("PUSHT_CKPTS", RT + "/run_pure/checkpoints").split(",")
N = int(os.environ.get("PUSHT_N", "20"))

os.environ["PUSHT_CHUNK"] = "1"
sys.argv = ["ev", "arx_goai_12task_r0"]
for p in (OFT, OFT + "/vla-scripts"):
    sys.path.insert(0, p)
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register("openvla", PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
from peft import PeftModel
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, STOP_INDEX, ACTION_DIM, IGNORE_INDEX

BINS = 256
BIN_CENTERS = np.linspace(-1.0, 1.0, BINS).astype(np.float32)
nb = np.load(BOUNDS); amin, amax = nb["amin"], nb["amax"]
def norm_bounds(a): return np.clip(2.0*(a-amin)/(amax-amin+1e-8)-1.0, -1, 1).astype(np.float32)
def c14(h, g):
    return np.concatenate([h[g+"/left_arm_joint_states"][:], h[g+"/left_ee_joint_states"][:],
                           h[g+"/right_arm_joint_states"][:], h[g+"/right_ee_joint_states"][:]], axis=1).astype(np.float32)
def decode(buf): return cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()

meta = json.load(open(os.path.dirname(CKPTS[0]) + "/../run_meta.json")) if os.path.exists(os.path.dirname(CKPTS[0]) + "/../run_meta.json") else None
train_eps = json.load(open(SPLIT))["train"]; val_eps = json.load(open(SPLIT))["val"]
# q01/q99 from the training split (same as training-time tokenizer)
A = []
for e in train_eps:
    with h5py.File(os.path.join(HDF5_DIR, e), "r") as h: A.append(c14(h, "action"))
A = np.concatenate(A, 0)
q01 = np.quantile(A, 0.01, axis=0).astype(np.float32); q99 = np.quantile(A, 0.99, axis=0).astype(np.float32)
def detok(tok):
    c = BIN_CENTERS[np.clip(tok, 0, BINS-1)]
    return (c + 1.0)/2.0*(q99-q01+1e-8) + q01

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
base = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                             trust_remote_code=True).to("cuda")
base.vision_backbone.set_num_images_in_input(1)
instr = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
INSTR_TEXT = "What action should the robot take to " + instr + "?"

def prompt_ids(dev="cuda"):
    b = PurePromptBuilder("openvla"); b.add_turn("human", INSTR_TEXT)
    ids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to(dev)
    if int(ids[0, -1].item()) != 29871:
        ids = torch.cat([ids, torch.tensor([[29871]], dtype=ids.dtype, device=dev)], dim=1)
    return ids

@torch.no_grad()
def predict_autoreg(vla, pv):
    pids = prompt_ids(); plen = pids.shape[1]; toks = []
    for k in range(ACTION_DIM):
        if toks:
            aid = torch.tensor([[ACTION_TOKEN_BEGIN_IDX + t + 1 for t in toks]], dtype=pids.dtype, device="cuda")
            inp = torch.cat([pids, aid], dim=1)
        else:
            inp = pids
        lab = inp.clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv, labels=lab)
        lg = out.logits[0, -1]
        cand = lg[ACTION_TOKEN_BEGIN_IDX+1: ACTION_TOKEN_BEGIN_IDX+1+BINS]
        toks.append(int(torch.argmax(cand).item()))
    return np.array(toks)

@torch.no_grad()
def teacher_forced(vla, pv, gt_tok):
    pids = prompt_ids(); plen = pids.shape[1]
    aid = torch.tensor([[ACTION_TOKEN_BEGIN_IDX + int(t) + 1 for t in gt_tok]], dtype=pids.dtype, device="cuda")
    stop = torch.full((1,1), STOP_INDEX, dtype=pids.dtype, device="cuda")
    inp = torch.cat([pids, aid, stop], dim=1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv, labels=inp.clone())
    lg = out.logits[0]
    preds = []
    for k in range(ACTION_DIM):
        c = lg[plen + k - 1][ACTION_TOKEN_BEGIN_IDX+1: ACTION_TOKEN_BEGIN_IDX+1+BINS]
        preds.append(int(torch.argmax(c).item()))
    return np.array(preds)

_fixp = os.environ.get("PUSHT_SAMPLES", RT + "/fixed_samples.json")
_fix = json.load(open(_fixp))
samples = []
for e, t in _fix[:N]:
    with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
        img = proc.image_processor.apply_transform(Image.fromarray(decode(h["vision/cam_head/colors"][t])))
        pv = img.unsqueeze(0).to(torch.bfloat16).to("cuda")
        gt_raw = c14(h, "action")[t].copy()
    samples.append((e, t, pv, gt_raw))
print("EVAL_SAMPLES %d from held-out %d episodes" % (len(samples), len(val_eps)), flush=True)

for ck in CKPTS:
    ck = ck.strip()
    if not ck or not os.path.isdir(ck): continue
    vla = PeftModel.from_pretrained(base, ck + "/lora_adapter")
    vla.eval()
    accs, l1s_autoreg, l1s_tf = [], [], []
    preds_all, gts_all = [], []
    for e, t, pv, gt_raw in samples:
        gt_norm = norm_bounds(gt_raw)
        gt_tok = np.digitize(np.clip(2.0*(gt_raw-q01)/(q99-q01+1e-8)-1.0, -1, 1), BIN_CENTERS)
        pt = predict_autoreg(vla, pv)
        tt = teacher_forced(vla, pv, gt_tok)
        accs.append(float((tt == gt_tok).mean()))
        l1s_autoreg.append(float(np.abs(norm_bounds(detok(pt)) - gt_norm).mean()))
        l1s_tf.append(float(np.abs(norm_bounds(detok(tt)) - gt_norm).mean()))
        preds_all.append(norm_bounds(detok(pt))); gts_all.append(gt_norm)
    P = np.stack(preds_all); G = np.stack(gts_all)
    uniq = len({tuple(np.round(r, 3)) for r in P})
    print("RESULT %-14s tf_acc=%.4f autoreg_L1=%.4f tf_L1=%.4f | pred_std=%.4f gt_std=%.4f ratio=%.3f uniq_preds=%d/%d" % (
        os.path.basename(ck), float(np.mean(accs)), float(np.mean(l1s_autoreg)), float(np.mean(l1s_tf)),
        float(P.std(0).mean()), float(G.std(0).mean()), float(P.std(0).mean()/(G.std(0).mean()+1e-9)), uniq, len(P)), flush=True)
    del vla
    torch.cuda.empty_cache()
print("PURE_EVAL_DONE")
