#!/usr/bin/env python3
"""R45 CPU-only control-plane trainer for the joint_delta_v1 contract.

Real OpenVLA training remains exclusively in training/r44_train.py.
This is a new entry point, not a renamed Push-T script. It is model-agnostic at the seam
(--model fake builds a 3-layer toy VLA used by the CPU tests), so the checkpoint/resume,
sampling and coverage logic can be verified without a GPU. The real model is built by
build_real_model() which is the only part that needs CUDA.

Every checkpoint is self-describing: weights, optimizer, scheduler, RNG (python/numpy/torch),
epoch, cursor into the epoch order, the order hash, global step, contract and normalizer ids.
resume_checkpoint() refuses to continue unless the recomputed order matches the stored hash and
the cursor points at exactly the sample that was next.
"""
import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import r45_data as DATA          # noqa: E402
import r44_delta_v1 as D         # noqa: E402

STOP = {"requested": False, "reason": None}


def _sig(signum, _frame):
    STOP["requested"] = True
    STOP["reason"] = "signal %d" % signum


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


# --------------------------------------------------------------------------------- model seam
class TinyVLA:
    """CPU stand-in with the same call surface as the real model (used by the tests)."""

    def __init__(self, dim=8, vocab=32000, n_action=14):
        import torch
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab, dim)
                self.proj = nn.Linear(DATA.N42, dim)
                self.out = nn.Linear(dim, n_action * 256)

            def forward(self, input_ids=None, pixel_values=None, proprio=None,
                        labels=None, **_kw):
                h = self.emb(input_ids).sum(dim=1) + self.proj(proprio)
                return self.out(h).reshape(-1, n_action, 256)

        self.torch = torch
        self.net = _M()
        self.proprio_projector = self.net.proj
        self.trainable = list(self.net.parameters())

    def parameters(self):
        return iter(self.trainable)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def loss_fn(self, out, target_bins):
        import torch
        return torch.nn.functional.cross_entropy(
            out.reshape(-1, 256), torch.as_tensor(target_bins).reshape(-1))


def build_model(kind, cfg):
    if kind == "fake":
        return TinyVLA()
    raise SystemExit("real model build requires the approved GPU phase (not implemented here: "
                     "use --model fake for CPU contract tests)")


# ------------------------------------------------------------------------------- checkpoint
def _rng_state():
    import torch
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    import torch
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(path, model, opt, sched, meta):
    import torch
    os.makedirs(path, exist_ok=True)
    payload = {"model": model.state_dict(),
               "optimizer": opt.state_dict() if opt else None,
               "scheduler": sched.state_dict() if sched else None,
               "rng": _rng_state(),
               "meta": meta}
    torch.save(payload, os.path.join(path, "state.pt"))
    save_json(os.path.join(path, "meta.json"), meta)
    return sha256_file(os.path.join(path, "state.pt"))


def load_checkpoint(path):
    import torch
    payload = torch.load(os.path.join(path, "state.pt"), map_location="cpu", weights_only=False)
    meta_path = os.path.join(path, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            disk = json.load(fh)
        if disk != payload["meta"]:
            raise SystemExit("BLOCKED: meta.json disagrees with state.pt in %s "
                             "(checkpoint is internally inconsistent)" % path)
    return payload


def verify_resume(payload, entries, base_seed):
    """Refuse to resume unless the stored order and cursor still describe the same run."""
    meta = payload["meta"]
    order = DATA.EpochOrder(len(entries), base_seed, meta["epoch"])
    if order.hash() != meta["order_hash"]:
        raise SystemExit("BLOCKED: epoch order hash changed (%s != %s)"
                         % (order.hash(), meta["order_hash"]))
    nxt = int(order.order[meta["cursor"]]) if meta["cursor"] < len(order.order) else None
    if nxt != meta["next_index"]:
        raise SystemExit("BLOCKED: cursor does not point at the recorded next sample")
    if meta["cursor"] > 0 and int(order.order[meta["cursor"] - 1]) != meta["last_index"]:
        raise SystemExit("BLOCKED: last sample index does not match the recorded cursor")
    if meta.get("phase") == "epoch_boundary" and meta["cursor"] != 0:
        raise SystemExit("BLOCKED: epoch-boundary checkpoint must have cursor 0")
    return order


# ---------------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fake", choices=("fake", "real"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "data", "m2_splits.json"))
    ap.add_argument("--hdf5-root", default="/home/T7/ojh/robot_actual/data/data/hdf5")
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="LoRA learning rate; the projector group uses 0.1x this")
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="CPU tests only: truncate the index (never used for real training)")
    ap.add_argument("--max-steps", type=int, default=0, help="hard safety stop")
    ap.add_argument("--checkpoint-every-frac", type=float, default=0.25)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--task-subset", default=None, help="debug only: comma separated task names")
    args = ap.parse_args()
    STOP["requested"], STOP["reason"] = False, None
    if args.micro_batch < 1 or args.grad_accum < 1 or args.epochs < 1:
        raise SystemExit("micro-batch, grad-accum and epochs must be positive")

    if args.limit_samples or args.task_subset:
        print("WARNING: debug options are set; this run is NOT a full-data training run")
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    os.makedirs(args.out, exist_ok=True)

    payload = json.load(open(args.normalizer))
    D.validate_normalizer(payload)
    splits = json.load(open(args.splits))["splits"]
    tasks = args.task_subset.split(",") if args.task_subset else None
    entries = DATA.build_index(splits, args.hdf5_root, "train", tasks=tasks)
    if args.limit_samples:
        entries = entries[: args.limit_samples]
    manifest = DATA.index_manifest(entries)
    save_json(os.path.join(args.out, "dataset_manifest.json"), manifest)

    ds = DATA.R45Dataset(entries, decode=(args.model == "real"))
    # This is the CPU control-plane harness. DATA_loader is not consumed here;
    # the genuine OpenVLA training path is training/r44_train.py.
    import torch
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    model = build_model(args.model, None)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    proj_params = list(model.proprio_projector.parameters()) if args.model == "real" else []
    groups = [{"params": lora_params, "lr": args.lr, "name": "lora"}]
    if proj_params:
        groups.append({"params": proj_params, "lr": args.lr * 0.1, "name": "projector"})
    import torch
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    steps_per_epoch = max(1, (len(entries) + args.micro_batch * args.grad_accum - 1)
                          // (args.micro_batch * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) *
        max(0.0, 0.5 * (1 + np.cos(np.pi * min(1.0, max(0, s - warmup) /
                                           max(1, total_steps - warmup))))))

    start_epoch, cursor, global_step = 0, 0, 0
    if args.resume:
        payload_ck = load_checkpoint(args.resume)
        verify_resume(payload_ck, entries, args.base_seed)
        if (int(payload_ck["meta"].get("micro_batch", -1)) != args.micro_batch
                or int(payload_ck["meta"].get("grad_accum", -1)) != args.grad_accum
                or payload_ck["meta"].get("index_sha256") != manifest["index_sha256"]
                or payload_ck["meta"].get("normalizer_sha256") != sha256_file(args.normalizer)):
            raise SystemExit("BLOCKED: checkpoint dataset, normalizer or batch contract differs")
        model.load_state_dict(payload_ck["model"])
        if payload_ck["optimizer"]:
            opt.load_state_dict(payload_ck["optimizer"])
        if payload_ck["scheduler"]:
            sched.load_state_dict(payload_ck["scheduler"])
        _set_rng_state(payload_ck["rng"])
        start_epoch = payload_ck["meta"]["epoch"]
        cursor = payload_ck["meta"]["cursor"]
        global_step = payload_ck["meta"]["global_step"]

    metrics_path = os.path.join(args.out, "train_metrics.jsonl")
    mf = open(metrics_path, "a", encoding="utf-8")
    t0 = time.time()
    seen = start_epoch * len(entries) + cursor
    for epoch in range(start_epoch, args.epochs):
        order = DATA.EpochOrder(len(entries), args.base_seed, epoch)
        idxs = order.order
        model.train()
        committed_cursor = cursor
        committed_rng = _rng_state()
        for j in range(cursor, len(idxs), args.micro_batch):
            if STOP["requested"]:
                # Any in-flight accumulation has not mutated optimizer weights.
                # Replay it from the last committed update on resume.
                opt.zero_grad(set_to_none=True)
                _set_rng_state(committed_rng)
                seen = epoch * len(entries) + committed_cursor
                break
            batch = [ds[int(idxs[k])] for k in range(j, min(j + args.micro_batch, len(idxs)))]
            group_start = (j // (args.micro_batch * args.grad_accum)) * (args.micro_batch * args.grad_accum)
            group_end = min(len(idxs), group_start + args.micro_batch * args.grad_accum)
            group_samples = group_end - group_start
            if args.model == "fake":
                # toy path: build a 42D vector from the observed history only (state, v, acc)
                pr = np.stack([
                    np.concatenate([b["history"][-1],
                                    b["history"][-1] - b["history"][-2],
                                    (b["history"][-1] - b["history"][-2])
                                    - (b["history"][-2] - b["history"][-3])])
                    for b in batch]).astype(np.float32)
                out = model.net(torch.tensor([[1, 2, 3]]), None, torch.tensor(pr))
                target = (np.array([b["index"] for b in batch], np.int64).reshape(-1, 1)
                          * np.ones((1, 14), np.int64)) % 256
                loss = model.loss_fn(out, target)
            else:
                loss = torch.tensor(0.0, requires_grad=True)
            (loss * (len(batch) / group_samples)).backward()
            if j + len(batch) >= group_end:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                committed_cursor = j + len(batch)
                committed_rng = _rng_state()
            seen += len(batch)
            if global_step % 10 == 0:
                rec = {"utc": utc_now(), "epoch": epoch, "step": global_step,
                       "cursor": j + len(batch), "committed_cursor": committed_cursor,
                       "loss": float(loss.detach()),
                       "samples_seen": seen,
                       "samples_per_s": round(seen / max(1e-9, time.time() - t0), 3),
                       "lr_lora": opt.param_groups[0]["lr"]}
                mf.write(json.dumps(rec, sort_keys=True) + "\n")
                mf.flush()
            if args.max_steps and global_step >= args.max_steps:
                STOP["requested"] = True
                STOP["reason"] = "max-steps"
                break
            if (committed_cursor == j + len(batch) and
                    (committed_cursor % max(1, int(len(idxs) * args.checkpoint_every_frac))
                     < args.micro_batch or committed_cursor == len(idxs))):
                _ckpt(args.out, model, opt, sched, epoch, committed_cursor, global_step,
                      order, entries, args, "periodic", phase="mid_epoch")
        if STOP["requested"]:
            # Keep the last durable optimizer-update boundary, never the epoch end.
            if committed_cursor < len(idxs):
                opt.zero_grad(set_to_none=True)
                _set_rng_state(committed_rng)
            seen = epoch * len(entries) + committed_cursor
            _ckpt(args.out, model, opt, sched, epoch, committed_cursor, global_step, order, entries,
                  args, "stopped", phase="mid_epoch")
            break
        nxt_order = DATA.EpochOrder(len(entries), args.base_seed, epoch + 1)
        _ckpt(args.out, model, opt, sched, epoch + 1, 0, global_step, nxt_order, entries, args,
              "epoch_end", phase="epoch_boundary")
        mf.write(json.dumps({"utc": utc_now(), "event": "epoch_end", "epoch": epoch,
                             "samples_seen": seen,
                             "wall_s": round(time.time() - t0, 1)}, sort_keys=True) + "\n")
        mf.flush()
        cursor = 0
    mf.close()
    save_json(os.path.join(args.out, "run_summary.json"),
              {"utc": utc_now(), "epochs_requested": args.epochs, "global_step": global_step,
               "samples_seen": seen, "stop_reason": STOP["reason"],
               "wall_s": round(time.time() - t0, 1), "n_train": len(entries),
               "index_sha256": manifest["index_sha256"]})
    return 0


def _ckpt(out, model, opt, sched, epoch, cursor, global_step, order, entries, args, kind,
          phase="mid_epoch"):
    meta = {"utc": utc_now(), "epoch": int(epoch), "cursor": int(cursor), "phase": phase,
            "order_hash": order.hash(), "global_step": int(global_step), "kind": kind,
            "base_seed": int(args.base_seed), "n_train": len(entries),
            "next_index": int(order.order[cursor]) if cursor < len(order.order) else None,
            "last_index": int(order.order[cursor - 1]) if cursor else None,
            "contract": D.CONTRACT_ID, "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "index_sha256": DATA.index_manifest(entries)["index_sha256"],
            "normalizer_sha256": sha256_file(args.normalizer)}
    d = os.path.join(out, "checkpoints", "epoch%03d_cursor%06d_%s" % (epoch, cursor, kind))
    sha = save_checkpoint(d, model, opt, sched, meta)
    print("[ckpt] %s state_sha256=%s step=%d" % (d, sha[:16], global_step), flush=True)


def DATA_loader(ds, workers, batch):
    """Index-style loader: the order comes from EpochOrder, never from shuffle=True."""
    import torch
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False,
        num_workers=max(0, int(workers)),
        worker_init_fn=DATA.worker_init,
        persistent_workers=bool(workers) > 0, prefetch_factor=(4 if workers else None),
        pin_memory=False)


if __name__ == "__main__":
    sys.exit(main())
