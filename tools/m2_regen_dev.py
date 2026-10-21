#!/usr/bin/env python3
"""Regenerate the M2 DEV predictions with the CORRECTED action-window alignment (issue #26 item 2).

The step2000 checkpoint is kept exactly as trained: this tool only re-runs DEV inference so the P6 gate
sees action-window rows instead of vision-token rows. It never trains and never writes into the
checkpoint directory; the new archive is a NEW file (the legacy one stays on disk, invalid for P6).

Real-forward verification performed while doing so (the "new output agrees with the teacher-forced text
alignment position by position" requirement):
  * pad = logits_len - len(labels) is measured from the ACTUAL forward for every prompt;
  * the 14 rows the window reads are exactly the rows that predict the 14 supervised label positions;
  * the full-vocabulary argmax of each of those rows lands INSIDE the legal 31744..31999 window, which
    is the end-to-end statement that the alignment and the codec agree.
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

import m2_train_shared as M2      # noqa: E402
import m2_multitask_data as MD    # noqa: E402  (load_task_index / task_dir live here)
import r42_adapter as R           # noqa: E402
import r15_resume as RS           # noqa: E402
import r11_train_pushT_r42 as R11  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", default="push_T,stack_bowls")
    ap.add_argument("--normalizer", default="data/m2_task_normalizer_2task.json")
    ap.add_argument("--splits", default="data/m2_splits.json")
    ap.add_argument("--dev-frames-per-task", type=int, default=8)
    ap.add_argument("--cams", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    tasks = M2.parse_tasks(a.tasks)
    _doc, split_map, payload, splits_path, norm_path = M2.load_m2_artifacts(
        tasks, a.splits, a.normalizer)
    indexes = {t: MD.load_task_index(t) for t in tasks}
    dev_frames = {t: M2.dev_frames_for_task(indexes[t], split_map[t], t,
                                            a.dev_frames_per_task, a.seed) for t in tasks}

    R11.force_arx_platform()                      # BEFORE any prismatic import
    proc, vla, proj, modeling_sha = R11.build_model(a.cams)
    man, ts = RS.verify_and_load_manifest(a.checkpoint)
    # Weights only, and every file hash-checked against the manifest before it is applied.
    # load_full_state() also restores the optimizer, the scheduler, the sampler and the RNG, which
    # this tool neither needs nor may touch; using it here would have required fabricating sampler
    # frames just to satisfy a signature. The integrity guarantee that matters (do the bytes match
    # what the training run recorded) is the explicit sha256 check below.
    import torch
    from safetensors.torch import load_file
    from peft.utils.save_and_load import set_peft_model_state_dict
    import hashlib

    def _sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    adapter = os.path.join(a.checkpoint, "lora_adapter", "adapter_model.safetensors")
    proj_pt = os.path.join(a.checkpoint, "proprio_projector_42.pt")
    for path, key in ((adapter, "lora_adapter/adapter_model.safetensors"),
                      (proj_pt, "proprio_projector_42.pt")):
        want = man["files"].get(key)
        got = _sha(path)
        if want is not None and got != want:
            raise SystemExit("manifest hash mismatch for %s: %s != %s" % (key, got, want))
        print("[regen] verified %s sha256=%s" % (key, got[:16]))
    if ts.get("model_kind") != "peft":
        raise SystemExit("checkpoint model_kind %r is not peft" % (ts.get("model_kind"),))
    set_peft_model_state_dict(vla, load_file(adapter))
    proj.load_state_dict(torch.load(proj_pt, map_location="cpu"))
    vla.eval()
    proj.eval()
    prompt_ids = {t: M2._make_prompt(proc, t, False) for t in tasks}
    n_prompt = {t: len(prompt_ids[t]) for t in tasks}
    norm_sha = M2.TN.normalizer_sha256(payload)
    ck_norm = (ts.get("data_contract") or {}).get("normalizer_sha256")
    ck_tasks = sorted((ts.get("task_mix") or {}).get("tasks") or [])
    print("[regen] checkpoint global_step=%s normalizer=%s" % (ts.get("global_step"), ck_norm))
    if ck_norm and ck_norm != norm_sha:
        raise SystemExit("normalizer mismatch: checkpoint %s vs payload %s" % (ck_norm, norm_sha))
    if ck_tasks and ck_tasks != sorted(tasks):
        raise SystemExit("task set mismatch: checkpoint %s vs %s" % (ck_tasks, sorted(tasks)))

    checks = {"pad": {}, "in_window": {}, "labels_present": {}}

    def predict(prompt_task, frame_task, episode, t):
        import torch
        state_, action_ = M2._read_state_action(indexes[frame_task]["hdf5_dir"], episode)
        row = M2.row_for(payload, frame_task, state_, action_, t, episode=episode)
        lab = M2.labels_for(payload, frame_task, prompt_ids[prompt_task], row["residual"])
        pv = M2._episode_images(proc, episode, t, a.cams, indexes[frame_task]["hdf5_dir"])
        inp = torch.tensor([lab["input_ids"]], dtype=torch.long, device="cuda")
        pr = torch.from_numpy(row["features"]).reshape(1, 42).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # the R42 forward is the TRAINING forward (pure-autoreg with labels), so labels and the
            # custom 42D projector must be passed exactly as the trainer does; without them the model
            # refuses with "labels required for multimodal autoregressive forward"
            o = vla(input_ids=inp, attention_mask=torch.ones_like(inp), pixel_values=pv,
                    proprio=pr, proprio_projector=proj, labels=inp.clone())
        logits = o.logits.float().cpu().numpy()[0]
        labels = np.asarray(lab["labels"]).reshape(-1)
        pad = int(logits.shape[0]) - int(labels.shape[0])
        nprompt = int(n_prompt[prompt_task])
        rows = np.arange(pad + nprompt - 1, pad + nprompt - 1 + M2.N_ACTIONS)
        sup = np.arange(nprompt, nprompt + M2.N_ACTIONS)
        am = logits[rows].argmax(axis=1)
        checks["pad"].setdefault(prompt_task, set()).add(pad)
        checks["in_window"].setdefault(prompt_task, []).append(
            bool(((am >= R.TOKEN_BEGIN_IDX + 1) & (am <= R.TOKEN_BEGIN_IDX + 256)).all()))
        checks["labels_present"].setdefault(prompt_task, []).append(bool((labels[sup] != -100).all()))
        return {"logits": logits, "labels": labels, "n_prompt": nprompt,
                "residual": row["residual"]}

    blocks = M2.evaluate_dev_ce(predict, payload, tasks, dev_frames,
                               log_fn=lambda m: print("[regen] " + m, flush=True))
    extra = {"action_window_alignment": M2.ACTION_WINDOW_ALIGNMENT,
             "checkpoint_dir": os.path.abspath(a.checkpoint),
             "checkpoint_global_step": int(ts["global_step"]),
             "lora_sha256": man["files"].get("lora_adapter/adapter_model.safetensors"),
             "projector_sha256": man["files"]["proprio_projector_42.pt"],
             "normalizer_sha256": norm_sha, "modeling_sha256": modeling_sha,
             "dev_frames_sha256": {t: blocks[t]["frames_sha256"] for t in tasks},
             "prompt_ids_sha256": {t: M2.sha256_json(prompt_ids[t]) for t in tasks},
             "no_retraining": True, "regenerated_over_legacy_alignment": True}
    digest = M2.write_eval_archive(a.out, blocks, data_contract=None, extra=extra)

    pads = {k: sorted(v) for k, v in checks["pad"].items()}
    ok_in = all(all(v) for v in checks["in_window"].values())
    ok_lab = all(all(v) for v in checks["labels_present"].values())
    rep = {"archive": os.path.abspath(a.out), "archive_sha256": digest, "tasks": tasks,
           "ce_window": {t: blocks[t]["ce_mean"] for t in tasks},
           "ce_full": {t: blocks[t]["ce_full_mean"] for t in tasks},
           "vision_prefix_tokens": pads,
           "all_argmax_inside_legal_window": ok_in,
           "all_14_supervised_positions_present": ok_lab,
           "n_frames_per_task": {t: len(blocks[t]["frames"]) for t in tasks},
           "checkpoint_global_step": int(ts["global_step"]),
           "n_prompts": len(checks["pad"])}
    out_rep = a.report or (os.path.splitext(a.out)[0] + "_regen_report.json")
    with open(out_rep, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    if not (ok_in and ok_lab):
        raise SystemExit("alignment verification FAILED - this archive must not be used for P6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
