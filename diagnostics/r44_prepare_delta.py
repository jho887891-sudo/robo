#!/usr/bin/env python3
"""R44 preparation: verify the displacement data contract, fit joint_delta_v1 on TRAIN only,
prove epoch-complete coverage, and compute the real training budget.

This script never trains, never loads the 7B, never touches CUDA.  It produces the artifact set
R44 section 9 asks for, and it decides between "ready to launch" and "budget not approved".
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))
import r44_delta_v1 as D  # noqa: E402

TOP = ("left_arm_joint_states", "left_ee_joint_states",
       "right_arm_joint_states", "right_ee_joint_states")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def load_14d(path):
    """Read the 14D state/action blocks in the frozen joint order."""
    import h5py
    with h5py.File(path, "r") as h:
        out = {}
        for group in ("state", "action"):
            cols = [np.asarray(h[group][k], dtype=np.float64).reshape(len(h[group][k]), -1)
                    for k in TOP]
            out[group] = np.concatenate(cols, axis=1)
        T = len(out["state"])
        cams = [c for c in CAMERAS if ("vision/%s/colors" % c) in h]
        out["T"] = T
        out["cameras"] = cams
        out["instruction"] = str(np.asarray(h["instruction"][()]).item()) \
            if "instruction" in h else None
    if out["state"].shape[1] != 14 or out["action"].shape[1] != 14:
        raise D.R44ContractError("%s is not 14D (%s / %s)"
                                 % (path, out["state"].shape, out["action"].shape))
    return out


def scan(splits, root):
    manifest = {"tasks": {}, "totals": {"train_samples": 0, "dev_samples": 0}}
    deltas = {}
    max_contract_err = 0.0
    contract_violations = []
    for task, d in splits.items():
        block = {"train_episodes": 0, "dev_episodes": 0, "train_samples": 0,
                 "dev_samples": 0, "excluded": [], "frames_min": None, "frames_max": None}
        deltas[task] = []
        for which in ("train", "dev"):
            for rel in d[which]:
                path = os.path.join(root, task, "arx_x5", "data", os.path.basename(rel))
                if not os.path.isfile(path):
                    block["excluded"].append({"episode": rel, "reason": "file missing"})
                    continue
                ep = load_14d(path)
                if len(ep["cameras"]) != 3:
                    block["excluded"].append({"episode": rel, "reason": "cameras=%s" % ep["cameras"]})
                    continue
                # A_t == S_{t+1} on the whole episode
                err = float(np.max(np.abs(ep["action"][:-1] - ep["state"][1:])))
                max_contract_err = max(max_contract_err, err)
                if err > 0.0:
                    contract_violations.append({"episode": rel, "max_abs_err": err})
                idx = np.arange(2, ep["T"] - 1)
                n = int(len(idx))
                if which == "train":
                    block["train_episodes"] += 1
                    block["train_samples"] += n
                    deltas[task].append(ep["action"][idx] - ep["state"][idx])
                else:
                    block["dev_episodes"] += 1
                    block["dev_samples"] += n
                block["frames_max"] = max(block["frames_max"] or 0, ep["T"])
                block["frames_min"] = min(block["frames_min"] or 10 ** 9, ep["T"])
        block["train_samples"] = int(block["train_samples"])
        block["dev_samples"] = int(block["dev_samples"])
        manifest["tasks"][task] = block
        manifest["totals"]["train_samples"] += block["train_samples"]
        manifest["totals"]["dev_samples"] += block["dev_samples"]
    manifest["max_abs_action_vs_next_state"] = max_contract_err
    manifest["contract_violations"] = contract_violations
    return manifest, {k: np.concatenate(v, axis=0) for k, v in deltas.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default=os.path.join(ROOT, "data", "m2_splits.json"))
    ap.add_argument("--hdf5-root", default="/home/T7/ojh/robot_actual/data/data/hdf5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=2)
    ap.add_argument("--sec-per-step", type=float, default=3.9135,
                    help="measured wall clock per optimizer step (m2_two_2000: 7827 s / 2000)")
    ap.add_argument("--sec-per-dev-frame", type=float, default=0.36,
                    help="measured forward-only cost per DEV frame (R43 path A: 359 ms/token)")
    ap.add_argument("--gpu-hours-available", type=float, default=0.0,
                    help="approved GPU-hours for this run; 0 means no budget was approved")
    ap.add_argument("--roundtrip-samples", type=int, default=400)
    args = ap.parse_args()

    out = os.path.abspath(os.path.expanduser(args.out))
    if os.path.exists(out):
        print("BLOCKED: output directory exists: " + out, file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    report = {"schema": "r44-preflight/v1", "started_at_utc": utc_now(),
              "contract": D.CONTRACT_ID, "status": "RUNNING",
              "scope": {"training_started": False, "gpu_used": False,
                        "checkpoints_written": False, "simulator_or_robot": False},
              "checks": []}

    def check(name, ok, detail=None):
        report["checks"].append({"name": name, "result": "PASS" if ok else "FAIL",
                                 "detail": detail})
        print("[%s] %s %s" % ("PASS" if ok else "FAIL", name,
                              json.dumps(detail, sort_keys=True, default=str)[:300]), flush=True)
        return ok

    splits_doc = json.load(open(args.splits))
    splits = splits_doc["splits"]
    report["splits_path"] = args.splits
    report["splits_sha256"] = sha256_file(args.splits)

    manifest, deltas = scan(splits, args.hdf5_root)
    manifest["split_sha256"] = report["splits_sha256"]
    save_json(os.path.join(out, "dataset_manifest.json"), manifest)
    n_train = manifest["totals"]["train_samples"]
    n_dev = manifest["totals"]["dev_samples"]
    check("all_train_dev_episodes_14d_and_three_camera",
          not any(b["excluded"] for b in manifest["tasks"].values()),
          {k: len(v["excluded"]) for k, v in manifest["tasks"].items()})
    check("action_equals_next_state_bit_exact",
          manifest["max_abs_action_vs_next_state"] == 0.0,
          {"max_abs_err": manifest["max_abs_action_vs_next_state"],
           "violations": len(manifest["contract_violations"])})

    payload = D.fit_normalizer(deltas, split_sha256=report["splits_sha256"])
    D.validate_normalizer(payload)
    save_json(os.path.join(out, "normalizer.json"), payload)
    check("normalizer_fitted_on_train_only", True,
          {"tasks": {k: v["n_rows"] for k, v in payload["tasks"].items()},
           "sha256": payload["sha256"]})

    # codec round trip on real TRAIN rows (deterministic subset)
    rng = np.random.default_rng([args.base_seed, 44])
    rt = {}
    worst_norm = 0.0
    for task, d in deltas.items():
        q01, q99 = D.normalizer_for_task(payload, task)
        span = (q99 - q01).astype(np.float32) + 1e-8
        take = rng.choice(len(d), size=min(args.roundtrip_samples, len(d)), replace=False)
        err_n, err_p = [], []
        outside = 0
        total_elems = 0
        for i in take:
            delta = d[i]
            idx, tok = D.encode_delta(delta, task, payload)
            if tok.min() < D.LEGAL_TOKEN_MIN or tok.max() > D.LEGAL_TOKEN_MAX:
                raise D.R44ContractError("token outside the legal window")
            back = D.decode_delta(idx, task, payload)
            inside = (delta >= q01) & (delta <= q99)
            outside += int((~inside).sum())
            total_elems += int(inside.size)
            err = np.abs(back - delta)
            err_p.append(float(err.max()))
            if inside.any():
                err_n.append(float((err[inside] / span[inside]).max()))
        rt[task] = {"samples": int(len(take)),
                    "max_abs_delta_error_physical": max(err_p),
                    "mean_abs_delta_error_physical": float(np.mean(err_p)),
                    "max_in_range_error_normalised_units": max(err_n) if err_n else None,
                    "elements_outside_q01_q99_fraction": outside / max(1, total_elems)}
        worst_norm = max(worst_norm, max(err_n) if err_n else 0.0)
    half_step = 0.5 * D.MAX_NORM_QUANT_ERROR
    check("codec_roundtrip_within_half_codec_step_for_representable_values",
          worst_norm <= half_step + 1e-4,
          {"per_task": rt, "half_step_normalised_units": half_step,
           "note": ("values outside [q01,q99] are clipped by the [-1,1] normalisation and are "
                    "counted, not hidden")})

    # epoch-complete coverage proof over the real sample count
    epochs = []
    for ep in range(args.epochs):
        s = D.EpochSampler(n_train, args.base_seed, ep)
        cov = s.coverage()
        epochs.append({"epoch": ep, "order_hash": s.order_hash(),
                       "first_10": s.order[:10].tolist(), **cov})
        if not cov["complete"]:
            raise D.R44ContractError("epoch %d is not a complete permutation" % ep)
    check("epoch_complete_no_replacement_5_epochs",
          all(e["complete"] and e["unique_seen"] == n_train for e in epochs),
          {"epochs": len(epochs), "unique_seen": n_train, "min_visits": 1, "max_visits": 1})
    save_json(os.path.join(out, "sample_coverage.json"),
              {"n_train": n_train, "base_seed": args.base_seed, "epochs": epochs,
               "invariant": "unique_seen == total and min_visits == max_visits == 1"})

    steps_per_epoch = n_train
    total_steps = steps_per_epoch * args.epochs
    train_h = total_steps * args.sec_per_step / 3600.0
    min_h = steps_per_epoch * args.min_epochs * args.sec_per_step / 3600.0
    dev_h = n_dev * args.sec_per_dev_frame / 3600.0
    dev_total_h = dev_h * args.epochs
    ckpt_bytes = 2_200_000_000
    n_ckpt = args.epochs * 4
    budget = {
        "n_train_samples": n_train, "n_dev_samples": n_dev,
        "steps_per_epoch": steps_per_epoch, "max_epochs": args.epochs,
        "max_optimizer_steps": total_steps,
        "min_epochs": args.min_epochs,
        "min_optimizer_steps": steps_per_epoch * args.min_epochs,
        "sec_per_step": args.sec_per_step,
        "train_gpu_hours_max": train_h, "train_gpu_hours_min": min_h,
        "dev_eval_hours_per_epoch": dev_h, "dev_eval_hours_total": dev_total_h,
        "checkpoints": {"count": n_ckpt, "bytes_each": ckpt_bytes,
                        "total_gb": n_ckpt * ckpt_bytes / 1e9},
        "total_gpu_hours_max": train_h + dev_total_h,
        "gpu_hours_available": args.gpu_hours_available,
        "sufficient": args.gpu_hours_available >= train_h + dev_total_h,
        "note": ("sec_per_step is measured, not assumed: m2_two_2000_20261021T123314Z ran "
                 "2000 steps between 12:33:22Z and 14:43:49Z = 7827 s"),
    }
    save_json(os.path.join(out, "budget.json"), budget)
    save_json(os.path.join(out, "training_config.json"), {
        "target_representation": D.CONTRACT_ID,
        "target": D.TARGET_DESCRIPTION,
        "lora": {"r": 32, "lora_alpha": 32, "lora_dropout": 0.0,
                 "target_modules": ["q_proj", "v_proj"]},
        "optimizer": {"name": "AdamW", "weight_decay": 0.0, "grad_clip": 1.0,
                      "projector_lr_scale": 0.1},
        "sampler": {"type": "epoch_complete_no_replacement", "base_seed": args.base_seed},
        "epochs_max": args.epochs, "epochs_min": args.min_epochs,
        "dev_eval_every_epoch": True,
        "checkpoint_every_fraction_of_epoch": 0.25,
        "precision": "bf16",
        "normalizer_sha256": payload["sha256"],
        "n_train_samples": n_train,
    })

    dataset_hashes = {"splits_sha256": report["splits_sha256"],
                      "normalizer_sha256": payload["sha256"],
                      "episode_files": {}}
    for task, d in splits.items():
        rows = []
        for rel in d["train"] + d["dev"]:
            path = os.path.join(args.hdf5_root, task, "arx_x5", "data", os.path.basename(rel))
            if os.path.isfile(path):
                st = os.stat(path)
                rows.append({"episode": os.path.basename(rel), "size_bytes": st.st_size,
                             "mtime_ns": st.st_mtime_ns})
        dataset_hashes["episode_files"][task] = rows
    save_json(os.path.join(out, "dataset_hashes.json"), dataset_hashes)

    budget_ok = budget["sufficient"]
    check("gpu_budget_covers_the_approved_plan", budget_ok,
          {"required_gpu_hours": round(budget["total_gpu_hours_max"], 1),
           "approved_gpu_hours": args.gpu_hours_available})
    report["budget"] = budget
    report["status"] = "READY" if budget_ok else "BLOCKED_BUDGET"
    report["elapsed_s"] = time.time() - t0
    save_json(os.path.join(out, "report.json"), report)
    with open(os.path.join(out, "environment.txt"), "w", encoding="utf-8") as fh:
        fh.write("python %s\n" % sys.version.split()[0])
        fh.write("numpy %s\n" % np.__version__)
        fh.write("hdf5_root %s\n" % args.hdf5_root)
        fh.write("host %s\n" % os.uname().nodename)
    code = 0 if budget_ok else 2
    with open(os.path.join(out, "exit_code.txt"), "w", encoding="utf-8") as fh:
        fh.write(str(code) + "\n")
    names = sorted(n for n in os.listdir(out) if n != "SHA256SUMS")
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        for n in names:
            fh.write("%s  %s\n" % (sha256_file(os.path.join(out, n)), n))
    print("STATUS=%s steps/epoch=%d max_steps=%d required_gpu_hours=%.1f exit=%d"
          % (report["status"], steps_per_epoch, total_steps, budget["total_gpu_hours_max"], code))
    return code


if __name__ == "__main__":
    sys.exit(main())
