#!/usr/bin/env python3
"""OFT checkpoint eval on the SAME 20 held-out samples, reporting the first-step L1
(apples-to-apples with pure OpenVLA chunk=1) plus per-step breakdown and diversity."""
import os, sys, json, random
import numpy as np, torch, h5py, cv2
from PIL import Image

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
MODEL = os.environ.get("PUSHT_MODEL", RT + "/models_openvla-7b")
HDF5_DIR = os.environ.get("PUSHT_HDF5", RT + "/hdf5_push_T/arx_x5/data")
OFT = os.environ.get("PUSHT_OFT", RT + "/openvla_oft_src")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/run80_cont/split.json")
BOUNDS = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
CKPT = os.environ["PUSHT_CKPT"]
TAG = os.environ.get("PUSHT_INIT_TAG", "pushT")
N = int(os.environ.get("PUSHT_N", "20"))

sys.argv = ["oft", "arx_goai_12task_r0"]
for p in (OFT, OFT + "/vla-scripts"):
    sys.path.insert(0, p)
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
from peft import PeftModel
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM, IGNORE_INDEX, ACTION_TOKEN_BEGIN_IDX, STOP_INDEX

CAMS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
INSTR = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
nb = np.load(BOUNDS); amin, amax, pmin, pmax = nb["amin"], nb["amax"], nb["pmin"], nb["pmax"]
def na(a): return np.clip(2.0*(a-amin)/(amax-amin+1e-8)-1.0, -1, 1).astype(np.float32)
def npr(s): return np.clip(2.0*(s-pmin)/(pmax-pmin+1e-8)-1.0, -1, 1).astype(np.float32)
def c14(h, g):
    return np.concatenate([h[g+"/left_arm_joint_states"][:], h[g+"/left_ee_joint_states"][:],
                           h[g+"/right_arm_joint_states"][:], h[g+"/right_ee_joint_states"][:]], axis=1).astype(np.float32)
def decode(b): return cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                            trust_remote_code=True).to("cuda")
vla.vision_backbone.set_num_images_in_input(3)
vla = PeftModel.from_pretrained(vla, CKPT + "/lora_adapter")
inner = getattr(vla, "model", None) or getattr(getattr(vla, "base_model", None), "model", None)
film = FiLMedPrismaticVisionBackbone(vision_backbone=inner.vision_backbone, llm_dim=vla.llm_dim).to("cuda")
film.load_state_dict(torch.load(CKPT + "/vision_backbone--" + TAG + ".pt", map_location="cpu"))
inner.vision_backbone = film
proprio = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim).to("cuda")
proprio.load_state_dict(torch.load(CKPT + "/proprio_projector--" + TAG + ".pt", map_location="cpu"))
ah = L1RegressionActionHead(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=14).to("cuda")
ah.load_state_dict(torch.load(CKPT + "/action_head--" + TAG + ".pt", map_location="cpu"))
vla.eval(); film.eval(); proprio.eval(); ah.eval()
print("LOADED", CKPT, flush=True)
n_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + 1

_fix = json.load(open(os.environ.get("PUSHT_SAMPLES", RT + "/fixed_samples.json")))
samples = []
for e, t in _fix[:N]:
    with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
        imgs = [proc.image_processor.apply_transform(Image.fromarray(decode(h["vision/"+c+"/colors"][t]))) for c in CAMS]
        pv = torch.cat(imgs, dim=0).unsqueeze(0).to(torch.bfloat16).to("cuda")
        pr = torch.from_numpy(npr(c14(h, "state")[t])).unsqueeze(0).to("cuda")
        gt = torch.from_numpy(na(c14(h, "action")[t:t+NUM_ACTIONS_CHUNK])).unsqueeze(0).to("cuda")
    samples.append((pv, pr, gt))

step_l1 = np.zeros(NUM_ACTIONS_CHUNK); preds0 = []; gts0 = []; deltas = []; gtdeltas = []
with torch.no_grad():
    for pv, pr, gt in samples:
        b = PurePromptBuilder("openvla"); b.add_turn("human", "What action should the robot take to " + INSTR + "?")
        ids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids.to("cuda")
        if int(ids[0, -1].item()) != 29871:
            ids = torch.cat([ids, torch.tensor([[29871]], dtype=ids.dtype, device="cuda")], dim=1)
        plen = ids.shape[1]
        iid = torch.cat([ids, torch.ones((1, NUM_ACTIONS_CHUNK*ACTION_DIM), dtype=ids.dtype, device="cuda"),
                         torch.full((1,1), STOP_INDEX, dtype=ids.dtype, device="cuda")], dim=1)
        lab = torch.full_like(iid, IGNORE_INDEX, dtype=torch.long)
        lab[:, plen:plen+NUM_ACTIONS_CHUNK*ACTION_DIM] = ACTION_TOKEN_BEGIN_IDX + 1
        lab[:, -1] = STOP_INDEX
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=iid, attention_mask=torch.ones_like(iid), pixel_values=pv, labels=lab,
                      output_hidden_states=True, proprio=pr, proprio_projector=proprio, use_film=True)
            hs = out.hidden_states[-1]; s0 = n_patches + (plen - 1)
            pred = ah.predict_action(hs[:, s0:s0+NUM_ACTIONS_CHUNK*ACTION_DIM, :]).float()
        d = (pred[0].cpu().numpy() - gt[0].cpu().numpy())
        step_l1 += np.abs(d).mean(axis=1)
        preds0.append(pred[0,0].cpu().numpy()); gts0.append(gt[0,0].cpu().numpy())
        deltas.append(np.abs(np.diff(pred[0].cpu().numpy(), axis=0)).max())
        gtdeltas.append(np.abs(np.diff(gt[0].cpu().numpy(), axis=0)).max())
step_l1 /= len(samples)
P = np.stack(preds0); G = np.stack(gts0)
print("CHUNK_MEAN_L1 = %.4f" % step_l1.mean())
print("FIRST_STEP_L1 = %.4f   (comparable with pure OpenVLA chunk=1)" % step_l1[0])
print("PER_STEP_L1 = " + " ".join("%.3f" % v for v in step_l1))
print("STEP0 pred_std=%.4f gt_std=%.4f ratio=%.3f uniq=%d/%d" % (
    P.std(0).mean(), G.std(0).mean(), P.std(0).mean()/(G.std(0).mean()+1e-9),
    len({tuple(np.round(r,3)) for r in P}), len(P)))
print("WITHIN_CHUNK_MAXDELTA pred=%.4f gt=%.4f (collapse indicator)" % (np.mean(deltas), np.mean(gtdeltas)))
print("OFT_EVAL_DONE")
