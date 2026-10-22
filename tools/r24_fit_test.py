#!/usr/bin/env python3
"""R24 discriminating experiment: can the model fit the frames it TRAINED on?

H1 (under-trained), H3 (generalisation) and H4 (action representation) make different predictions about
the TRAIN frames the optimiser actually saw:

  H1 under-training      train error still high, DEV error high, both falling with steps
  H3 generalisation      train error much LOWER than DEV (the model memorised episodes)
  H4 representation      train error as bad as DEV even though the loss fell (the model cannot express
                         the mapping, so more data or more steps of the same recipe cannot help)

This tool measures the same quantities as m2_physical_audit.py on TRAIN frames instead of DEV frames,
with the identical codec and SAFE clip, and writes the archive the audit tool consumes. It performs
INFERENCE ONLY: no optimizer step, no checkpoint write. It is explicitly not a training run.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("training", "code/xpolicylab/policy/OpenVLA_Pure", "code/xpolicylab/policy/OpenVLA_R42"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m2_train_shared as M2       # noqa: E402
import m2_multitask_data as MD     # noqa: E402
import r42_adapter as R            # noqa: E402
import r15_resume as RS            # noqa: E402
import r11_train_pushT_r42 as R11  # noqa: E402
import r24_frame_manifest as FM    # noqa: E402


def load_verified(checkpoint, cams):
    import torch, hashlib
    from safetensors.torch import load_file
    from peft.utils.save_and_load import set_peft_model_state_dict

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    R11.force_arx_platform()
    proc, vla, proj, modeling_sha = R11.build_model(cams)
    man, ts = RS.verify_and_load_manifest(checkpoint)
    adapter = os.path.join(checkpoint, "lora_adapter", "adapter_model.safetensors")
    proj_pt = os.path.join(checkpoint, "proprio_projector_42.pt")
    for p, k in ((adapter, "lora_adapter/adapter_model.safetensors"),
                 (proj_pt, "proprio_projector_42.pt")):
        if man["files"].get(k) and sha(p) != man["files"][k]:
            raise SystemExit("manifest hash mismatch for %s" % k)
    set_peft_model_state_dict(vla, load_file(adapter))
    proj.load_state_dict(torch.load(proj_pt, map_location="cpu"))
    vla.eval(); proj.eval()
    return proc, vla, proj, ts, modeling_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True,
                    help="frozen frame manifest (r24_frame_manifest.py output): declares which frames "
                         "are TRAIN and which are DEV, with hashes, BEFORE any model is run")
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    man = json.load(open(a.manifest))
    if man.get("frozen_before_model_run") is not True:
        raise SystemExit("the frame manifest must be frozen BEFORE evaluation (frozen_before_model_run)")
    tasks = man["tasks"]
    _doc, split, payload, _sp, _np = M2.load_m2_artifacts(tasks, a.splits, a.normalizer)
    indexes = {t: MD.load_task_index(t) for t in tasks}
    proc, vla, proj, ts, modeling_sha = load_verified(a.checkpoint, a.cams)
    prompt_ids = {t: M2._make_prompt(proc, t, False) for t in tasks}
    lo = R.TOKEN_BEGIN_IDX + 1
    hi = lo + 256

    def window(lg):
        w = np.asarray(lg, np.float32).reshape(-1)[lo:hi]
        if w.size != 256:
            raise SystemExit("logits do not cover the 256 legal bins")
        return w

    arrays = {"action_window_alignment": np.array("vision-prefix-aware-v2"),
              "schema": np.array("r24-fit-archive-v1"),
              "checkpoint_dir": np.array(os.path.abspath(a.checkpoint)),
              "checkpoint_global_step": np.array(int(ts["global_step"])),
              "modeling_sha256": np.array(modeling_sha),
              "frame_manifest_sha256": np.array(man["manifest_sha256"]),
              "no_optimizer_step": np.array("true")}
    report = {"checkpoint": os.path.abspath(a.checkpoint),
              "checkpoint_global_step": int(ts["global_step"]),
              "frame_manifest_sha256": man["manifest_sha256"], "splits": {}}
    for t in tasks:
        for split_name in ("train", "dev"):
            frames = man["frames"][t][split_name]
            eps, ticks, pred, true, probs = [], [], [], [], []
            for rec in frames:
                ep, tick = rec["episode"], int(rec["t"])
                state_, action_ = M2._read_state_action(indexes[t]["hdf5_dir"], ep)
                row = M2.row_for(payload, t, state_, action_, tick, episode=ep)
                lab = M2.labels_for(payload, t, prompt_ids[t], row["residual"])
                pv = M2._episode_images(proc, ep, tick, a.cams, indexes[t]["hdf5_dir"])
                inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
                pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                            proprio=pr, proprio_projector=proj, labels=inp.clone())
                logits = o.logits[0].float().cpu().numpy()
                labels = np.asarray(lab["labels"]).reshape(-1)
                w = M2.action_window_probs(logits, int(lab["n_prompt"]),
                                           n_text_tokens=len(labels))
                true_bins = M2.bins_from_residual(payload, t, row["residual"])
                eps.append(ep); ticks.append(tick)
                pred.append(w["pred_bins"]); probs.append(w["pred_probs"]); true.append(true_bins)
            key = "%s|%s" % (t, split_name)
            arrays[key + "/episodes"] = np.array(eps, dtype=object)
            arrays[key + "/t"] = np.array(ticks, np.int64)
            arrays[key + "/pred_bins"] = np.stack(pred)
            arrays[key + "/pred_probs"] = np.stack(probs)
            arrays[key + "/bins_true"] = np.stack(true)
            arrays[key + "/ce"] = np.array([0.0] * len(eps))
            arrays[key + "/ce_window"] = np.array([
                M2.ce_window_from_probs(p, b) for p, b in zip(probs, true)])
            report["splits"][key] = {"n_frames": len(eps),
                                     "ce_window_mean": float(np.mean(arrays[key + "/ce_window"]))}
            print("[fit] %-22s n=%d ce_window=%.4f" % (key, len(eps),
                                                       report["splits"][key]["ce_window_mean"]),
                  flush=True)
    np.savez_compressed(a.out, **arrays)
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
