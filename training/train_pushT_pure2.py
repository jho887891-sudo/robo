#!/usr/bin/env python3
"""Pure OpenVLA (autoregressive action tokens) with configurable INPUTS.
--cams 1|3 and --proprio 0|1 let us isolate the contribution of the action-output
METHOD (autoregressive discrete tokens) from the OBSERVATION INPUTS (1 cam vs 3 cams + proprio)."""
import os, sys, json, time, random, argparse
import numpy as np, torch, h5py, cv2
from PIL import Image

RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
MODEL = os.environ.get("PUSHT_MODEL", RT + "/models_openvla-7b")
HDF5_DIR = os.environ.get("PUSHT_HDF5", RT + "/hdf5_push_T/arx_x5/data")
OFT = os.environ.get("PUSHT_OFT", RT + "/openvla_oft_src")
BOUNDS_FILE = os.environ.get("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
SPLIT = os.environ.get("PUSHT_SPLIT", RT + "/split80.json")
SAMPLE_FIX = os.environ.get("PUSHT_SAMPLES", RT + "/fixed_samples.json")
OUT = os.environ.get("PUSHT_OUT", RT + "/run_pure3c")
_REAL_ARGV = sys.argv[1:]
os.environ["PUSHT_CHUNK"] = "1"
sys.argv = ["pure2", "arx_goai_12task_r0"]
for p in (OFT, OFT + "/vla-scripts"):
    sys.path.insert(0, p)

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
from peft import LoraConfig, get_peft_model
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, STOP_INDEX, ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX

sys.argv = ["pure2"] + _REAL_ARGV
CAMS3 = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
INSTR = "Push the T-shaped block to align it precisely with the gray T-shaped pad."
BINS = 256
BIN_CENTERS = np.linspace(-1.0, 1.0, BINS).astype(np.float32)

def log(m): print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)
def c14(h, g):
    return np.concatenate([h[g+"/left_arm_joint_states"][:], h[g+"/left_ee_joint_states"][:],
                           h[g+"/right_arm_joint_states"][:], h[g+"/right_ee_joint_states"][:]], axis=1).astype(np.float32)
def decode(buf): return cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()
def norm_a(a, q01, q99): return np.clip(2.0*(a-q01)/(q99-q01+1e-8)-1.0, -1, 1).astype(np.float32)
def norm_pr(s, pmin, pmax): return np.clip(2.0*(s-pmin)/(pmax-pmin+1e-8)-1.0, -1, 1).astype(np.float32)
def detok(tokens, q01, q99):
    c = BIN_CENTERS[np.clip(tokens, 0, BINS-1)]
    return (c + 1.0)/2.0*(q99-q01+1e-8) + q01
def tok(a_norm): return np.digitize(np.clip(a_norm, -1, 1), BIN_CENTERS).astype(np.int64)

def action_stats(eps):
    A = []
    for e in eps:
        with h5py.File(os.path.join(HDF5_DIR, e), "r") as h: A.append(c14(h, "action"))
    A = np.concatenate(A, 0)
    return np.quantile(A, 0.01, axis=0).astype(np.float32), np.quantile(A, 0.99, axis=0).astype(np.float32)

def build(cams, use_proprio):
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                trust_remote_code=True).to("cuda")
    vla.vision_backbone.set_num_images_in_input(cams)
    cfg = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear", init_lora_weights="gaussian")
    vla = get_peft_model(vla, cfg)
    tot = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    pp = None
    if use_proprio:
        pp = ProprioProjector(proprio_dim=14, llm_dim=vla.llm_dim).to("cuda")
    log("cams=%d proprio=%s | LoRA trainable=%.1fM | proprio projector=%.1fM" % (
        cams, use_proprio, tot/1e6, (sum(p.numel() for p in pp.parameters())/1e6) if pp else 0.0))
    return proc, vla, pp

def trainable_list(vla, pp):
    ps = [p for p in vla.parameters() if p.requires_grad]
    if pp is not None: ps += list(pp.parameters())
    return ps

def obs(h, t, proc, cams, use_proprio, pmin, pmax):
    imgs = [proc.image_processor.apply_transform(Image.fromarray(decode(h["vision/"+cc+"/colors"][t]))) for cc in CAMS3[:cams]]
    pv = torch.cat(imgs, dim=0).unsqueeze(0).to(torch.bfloat16).to("cuda")
    pr = None
    if use_proprio:
        pr = torch.from_numpy(norm_pr(c14(h, "state")[t], pmin, pmax)).unsqueeze(0).to("cuda")
    return pv, pr

def make_ids(proc, a_tokens):
    b = PurePromptBuilder("openvla"); b.add_turn("human", "What action should the robot take to " + INSTR + "?")
    ids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids
    if int(ids[0, -1].item()) != 29871:
        ids = torch.cat([ids, torch.tensor([[29871]], dtype=ids.dtype)], dim=1)
    aid = torch.tensor([[ACTION_TOKEN_BEGIN_IDX + int(t) + 1 for t in a_tokens]], dtype=ids.dtype)
    stop = torch.full((1, 1), STOP_INDEX, dtype=ids.dtype)
    inp = torch.cat([ids, aid, stop], dim=1)
    lab = torch.cat([torch.full_like(ids, IGNORE_INDEX, dtype=torch.long), aid.long(), stop.long()], dim=1)
    return inp, lab

@torch.no_grad()
def evaluate(vla, pp, proc, cams, use_proprio, pmin, pmax, q01, q99, eps, n):
    vla.eval()
    if pp is not None: pp.eval()
    random.seed(11)
    preds, gts = [], []
    for (e, t) in EPS_FIXED[:n]:
        with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
            pv, pr = obs(h, t, proc, cams, use_proprio, pmin, pmax)
            gt_raw = c14(h, "action")[t].copy()
        b = PurePromptBuilder("openvla"); b.add_turn("human", "What action should the robot take to " + INSTR + "?")
        pids = proc.tokenizer(b.get_prompt(), return_tensors="pt").input_ids
        if int(pids[0, -1].item()) != 29871:
            pids = torch.cat([pids, torch.tensor([[29871]], dtype=pids.dtype)], dim=1)
        toks = []
        for k in range(ACTION_DIM):
            if toks:
                aid = torch.tensor([[ACTION_TOKEN_BEGIN_IDX + x + 1 for x in toks]], dtype=pids.dtype)
                inp = torch.cat([pids, aid], dim=1).to("cuda")
            else:
                inp = pids.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                          labels=inp.clone(), proprio=pr, proprio_projector=pp)
            lg = out.logits[0, -1]
            cand = lg[ACTION_TOKEN_BEGIN_IDX+1: ACTION_TOKEN_BEGIN_IDX+1+BINS]
            toks.append(int(torch.argmax(cand).item()))
        preds.append(norm_a(detok(np.array(toks), q01, q99), q01, q99))
        gts.append(norm_a(gt_raw, q01, q99))
    vla.train()
    if pp is not None: pp.train()
    P = np.stack(preds); G = np.stack(gts)
    return (float(np.abs(P-G).mean()), float(P.std(0).mean()), float(G.std(0).mean()),
            len({tuple(np.round(r, 3)) for r in P}), len(P))

EPS_FIXED = []
def main():
    global EPS_FIXED
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_n", type=int, default=20)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--proprio", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_ckpt", type=str, default="")
    ap.add_argument("--lr_milestone", type=int, default=1000)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(OUT + "/checkpoints", exist_ok=True)
    sj = json.load(open(SPLIT)); train_eps, val_eps = sj["train"], sj["val"]
    EPS_FIXED = [tuple(x) for x in json.load(open(SAMPLE_FIX))]
    log("train=%d val=%d | cams=%d proprio=%d | chunk=%d dim=%d | FIXED_SAMPLES=%d" % (
        len(train_eps), len(val_eps), args.cams, args.proprio, NUM_ACTIONS_CHUNK, ACTION_DIM, len(EPS_FIXED)))
    q01, q99 = action_stats(train_eps)
    nb = np.load(BOUNDS_FILE); pmin, pmax = nb["pmin"], nb["pmax"]
    proc, vla, pp = build(args.cams, bool(args.proprio))
    if args.init_ckpt:
        from safetensors.torch import load_file
        from peft.utils.save_and_load import set_peft_model_state_dict
        st = load_file(args.init_ckpt + "/lora_adapter/adapter_model.safetensors")
        set_peft_model_state_dict(vla, st)
        log("loaded LoRA from %s (%d tensors)" % (args.init_ckpt, len(st)))
        ppf = args.init_ckpt + "/proprio_projector--pure.pt"
        if pp is not None and os.path.exists(ppf):
            pp.load_state_dict(torch.load(ppf, map_location="cpu"))
            log("loaded proprio projector from %s" % ppf)
    ps = trainable_list(vla, pp)
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_milestone], gamma=0.1)
    vla.train()
    if pp is not None: pp.train()
    t0 = time.time(); step = 0
    while step < args.steps:
        e = random.choice(train_eps)
        with h5py.File(os.path.join(HDF5_DIR, e), "r") as h:
            T = h["action/left_arm_joint_states"].shape[0]; t = random.randrange(0, max(1, T-1))
            pv, pr = obs(h, t, proc, args.cams, bool(args.proprio), pmin, pmax)
            gt_tok = tok(norm_a(c14(h, "action")[t], q01, q99))
        inp, lab = make_ids(proc, gt_tok)
        inp = inp.to("cuda"); lab = lab.to("cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv, labels=lab,
                      proprio=pr, proprio_projector=pp)
            loss = out.loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step(); sched.step(); step += 1
        if step % args.log_every == 0 or step == 1:
            log("step=%d loss=%.4f lr=%.2e gnorm=%.2f mem=%.1fGB %.1fs" % (
                step, loss.item(), sched.get_last_lr()[0], float(gn), torch.cuda.max_memory_allocated()/1e9, time.time()-t0))
        if step % args.save_every == 0:
            d = OUT + "/checkpoints/step%d" % step; os.makedirs(d, exist_ok=True)
            vla.save_pretrained(d + "/lora_adapter")
            if pp is not None: torch.save(pp.state_dict(), d + "/proprio_projector--pure.pt")
            log("saved " + d)
            l1, psd, gsd, uq, nn = evaluate(vla, pp, proc, args.cams, bool(args.proprio), pmin, pmax, q01, q99, val_eps, args.eval_n)
            log("EVAL step=%d autoreg_L1=%.4f pred_std=%.4f gt_std=%.4f ratio=%.3f uniq=%d/%d" % (step, l1, psd, gsd, psd/(gsd+1e-9), uq, nn))
    l1, psd, gsd, uq, nn = evaluate(vla, pp, proc, args.cams, bool(args.proprio), pmin, pmax, q01, q99, val_eps, args.eval_n)
    log("FINAL_EVAL autoreg_L1=%.4f pred_std=%.4f gt_std=%.4f ratio=%.3f uniq=%d/%d" % (l1, psd, gsd, psd/(gsd+1e-9), uq, nn))
    d = OUT + "/checkpoints/final"; os.makedirs(d, exist_ok=True)
    vla.save_pretrained(d + "/lora_adapter")
    if pp is not None: torch.save(pp.state_dict(), d + "/proprio_projector--pure.pt")
    json.dump({"variant": "pure_openvla", "cams": args.cams, "proprio": args.proprio, "steps": args.steps,
               "lr": args.lr, "lora_r": 32, "target_modules": "all-linear", "num_bins": BINS,
               "heldout_autoreg_L1": l1, "q01": q01.tolist(), "q99": q99.tolist()},
              open(OUT + "/run_meta.json", "w"), indent=1)
    log("PURE2_TRAIN_DONE steps=%d" % step)

if __name__ == "__main__":
    main()
