#!/usr/bin/env python3
"""R9 Gate B trace: real HDF5 -> 42D [S,V,Acc] -> R -> frozen bins -> 14 action tokens.

Runs on CPU with ZERO optimizer steps and does NOT load OpenVLA-7B. The forward pass is a
declared MOCK head (the real 7B forward is PENDING_GPU and not authorized); everything
before and after the forward - loader, proprio history, normalisation, target, codec,
token/label construction, decode to a physical command - is the real production code,
including the untouched entrypoint module training/train_pushT_pure2.py.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.environ.get("PUSHT_RUNTIME", "/home/dgut/pusht_runtime")
os.environ.setdefault("PUSHT_MODEL", RT + "/model_cache/openvla-7b-pristine")
os.environ.setdefault("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
os.environ.setdefault("PUSHT_BOUNDS", RT + "/push_T_bounds.npz")
os.environ.setdefault("PUSHT_SPLIT", os.path.join(REPO, "data", "split80.json"))
os.environ.setdefault("PUSHT_SAMPLES", RT + "/fixed_samples.json")
os.environ.setdefault("PUSHT_OFT", "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")
for sub in ("training", "diagnostics", "code/xpolicylab/policy/OpenVLA_Pure"):
    sys.path.insert(0, os.path.join(REPO, sub))

import torch  # noqa: E402

RESULTS = []


def check(name, ok, detail=None, label=""):
    RESULTS.append({"check": name, "pass": bool(ok), "label": label,
                    "detail": detail if detail is not None else {}})
    payload = json.dumps(detail, default=str)[:400] if detail else ""
    print("[%s] %-46s %-16s %s" % ("PASS" if ok else "FAIL", name, label, payload), flush=True)
    return bool(ok)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cuda_lines(func):
    """Report which lines of a real entrypoint function require CUDA (PENDING_GPU)."""
    try:
        src, start = inspect.getsourcelines(func)
    except OSError:
        return []
    return [{"line": start + i, "text": line.strip()}
            for i, line in enumerate(src) if "cuda" in line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import h5py
    import train_pushT_pure2 as TP
    import action_codec as AC
    import action_target as AT
    import r9_trainer_contract as RC
    import r8_vla_adapter as A

    contract, meta = RC.load_frozen_contract(args.contract)
    q01 = np.asarray(meta["q01"], np.float32)
    q99 = np.asarray(meta["q99"], np.float32)
    splits = RC.select_splits(os.environ["PUSHT_SPLIT"])
    hdf5_dir = os.environ["PUSHT_HDF5"]
    bounds = os.environ["PUSHT_BOUNDS"]
    nb = np.load(bounds)
    ae, ac, gs = nb["amin"].astype(np.float32), nb["amax"].astype(np.float32), None
    trace = {"contract": os.path.abspath(args.contract), "contract_sha256": sha256(args.contract),
             "entrypoint_module": os.path.abspath(TP.__file__),
             "entrypoint_sha256": sha256(os.path.abspath(TP.__file__)),
             "splits": {k: len(v) for k, v in splits.items()},
             "cuda_used": False, "optimizer_steps": 0, "seven_b_loaded": False}

    # ---------------------------------------------------------------- B1 import
    check("B1_real_entrypoint_imported",
          os.path.isfile(os.path.abspath(TP.__file__)) and TP.INSTR.startswith("Push the T"),
          {"module": TP.__file__, "sha256": trace["entrypoint_sha256"][:16],
           "cams3": TP.CAMS3, "action_dim_constant": AC.N_BINS},
          label="VERIFIED_CPU")

    # ------------------------------------------------- B2 legacy 14D path intact
    legacy_eps = splits["train"][:args.episodes]
    lq01, lq99 = TP.action_stats(legacy_eps, target_mode=AT.ABSOLUTE)
    with h5py.File(os.path.join(hdf5_dir, legacy_eps[0]), "r") as h:
        st = TP.c14(h, "state")
        pr14 = TP.norm_pr(st[40], nb["pmin"], nb["pmax"])
    degenerate = [int(i) for i in np.where(~(np.asarray(lq99) > np.asarray(lq01)))[0]]
    check("B2_legacy_14d_path_unchanged",
          np.asarray(lq01).shape == (14,) and np.asarray(lq99).shape == (14,)
          and np.asarray(pr14).shape == (14,)
          and bool(np.isfinite(lq01).all()) and bool(np.isfinite(lq99).all())
          and bool((np.asarray(lq99) >= np.asarray(lq01)).all()),
          {"q01_shape": list(np.asarray(lq01).shape), "pr14_shape": list(np.asarray(pr14).shape),
           "degenerate_dims_q99_eq_q01": degenerate,
           "episodes_used": legacy_eps,
           "note": "trainer still computes 80-episode quantiles; degeneracy here is a "
                   "small-sample artefact of using 2 episodes in this trace"},
          label="VERIFIED_CPU")

    # --------------------------------------- B3 real prompt / label construction
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(os.environ["PUSHT_MODEL"], trust_remote_code=True)
    with h5py.File(os.path.join(hdf5_dir, splits["train"][0]), "r") as h:
        state_ep = TP.c14(h, "state")
        action_ep = TP.c14(h, "action")
    t0 = 40
    v0 = state_ep[t0] - state_ep[t0 - 1]
    r0 = RC.target_residual(state_ep[t0], action_ep[t0], v0)
    idx0, ids0 = RC.encode_tokens(r0, q01, q99)
    inp, lab = TP.make_ids(proc, idx0)
    supervised = int((lab != -100).sum().item())
    legal_rc = AC.assert_legal_token_ids(np.asarray(inp[0, -15:-1].tolist()))
    legal = legal_rc is None or legal_rc is True
    check("B3_prompt_and_label_mask",
          supervised == 15 and legal and int(inp[0, -1].item()) == AC.STOP_INDEX
          and int(ids0.min()) >= 31744 and int(ids0.max()) <= 31999,
          {"n_supervised": supervised, "n_prompt_tokens": int(inp.shape[1]) - 15,
           "stop_token": int(inp[0, -1].item()), "token_range": [int(ids0.min()), int(ids0.max())],
           "prompt_ignored": int((lab[0, :inp.shape[1] - 15] == -100).sum().item())},
          label="VERIFIED_CPU")

    # ----------------------------------------------- B4 42D features, real HDF5
    feat_rows = []
    worst_delta = 0.0
    for name in splits["train"][:args.episodes]:
        with h5py.File(os.path.join(hdf5_dir, name), "r") as h:
            st = TP.c14(h, "state")
        for t in RC.frame_pool(len(st))[:args.frames]:
            f = RC.proprio42(st[t], st[t - 1], st[t - 2], contract)
            # independent recomputation: S from position bounds, V/Acc from signed motion scale
            s_n = np.clip(2.0 * (st[t] - nb["pmin"]) / (nb["pmax"] - nb["pmin"] + 1e-8) - 1.0, -1, 1)
            v = (st[t] - st[t - 1]) / meta["v_scale"]
            acc = ((st[t] - st[t - 1]) - (st[t - 1] - st[t - 2])) / meta["a_scale"]
            manual = np.concatenate([s_n, np.clip(v, -1, 1), np.clip(acc, -1, 1)]).astype(np.float32)
            worst_delta = max(worst_delta, float(np.abs(f - manual).max()))
            feat_rows.append({"episode": name, "t": int(t), "shape": list(f.shape),
                              "finite": bool(np.isfinite(f).all()),
                              "max_abs": float(np.abs(f).max())})
    check("B4_42d_features_real_hdf5",
          bool(feat_rows) and all(r["shape"] == [42] and r["finite"] and r["max_abs"] <= 1.0 + 1e-6
                                  for r in feat_rows) and worst_delta <= 1e-6,
          {"n_rows": len(feat_rows), "worst_delta_vs_manual": worst_delta, "samples": feat_rows[:3]},
          label="VERIFIED_CPU")

    # ------------------------------------------------------- B5 target identity
    r_at = AT.training_target(state_ep[t0], action_ep[t0], AT.RESIDUAL)
    r_manual = r_at - v0
    check("B5_target_R_identity",
          float(np.abs(r0 - r_manual).max()) <= 1e-6,
          {"max_abs_delta": float(np.abs(r0 - r_manual).max()),
           "R_abs_max": float(np.abs(r0).max()),
           "q01_head": q01[:3].tolist(), "q99_head": q99[:3].tolist()},
          label="VERIFIED_CPU")

    # ------------------------------------------------------- B6 decode round trip
    cmd, decoded, excluded = RC.decode_command(ids0, q01, q99, state_ep[t0], v0, bounds)
    step = (q99 - q01) / 255.0
    err = np.abs(cmd - action_ep[t0])
    clip_mask = (r0 >= q01) & (r0 <= q99)
    worst_in = float(err[clip_mask].max()) if clip_mask.any() else 0.0
    cv = state_ep[t0] + v0
    grip_ok = bool(np.allclose(cmd[[6, 13]], np.clip(cv[[6, 13]], ae[[6, 13]], ac[[6, 13]]), atol=1e-6))
    check("B6_decode_to_command",
          worst_in <= float(step.max()) + 1e-5 and grip_ok and excluded == [6, 13]
          and bool(np.isfinite(cmd).all()) and bool((cmd >= ae - 1e-6).all())
          and bool((cmd <= ac + 1e-6).all()),
          {"max_abs_error_in_range": worst_in, "one_bin_step_max": float(step.max()),
           "out_of_range_dims": [int(i) for i in np.where(~clip_mask)[0]],
           "gripper_from_S_plus_V": grip_ok, "excluded_dims": excluded,
           "physical_L1_this_frame": float(err.mean())},
          label="VERIFIED_CPU")

    # ------------------------------------------- B7 zero-update loss trace (MOCK)
    torch.manual_seed(0)
    hidden, vocab_window = 64, 256

    class MockActionHead(torch.nn.Module):
        """MOCK forward: stands in for OpenVLA-7B so the loss/label plumbing can be traced."""

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(42, hidden)
            self.head = torch.nn.Linear(hidden, vocab_window)

        def forward(self, proprio, action_labels):
            z = self.head(torch.tanh(self.proj(proprio)))
            logits = z.repeat(int(action_labels.shape[0]), 1)
            return logits

    head = MockActionHead()
    labels14 = torch.as_tensor(idx0, dtype=torch.long)
    feat = torch.as_tensor(RC.proprio42(state_ep[t0], state_ep[t0 - 1], state_ep[t0 - 2], contract))
    logits = head(feat.unsqueeze(0), labels14)
    loss = torch.nn.functional.cross_entropy(logits, labels14)
    loss.backward()
    grad_norm = float(sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None))
    check("B7_zero_update_loss_trace",
          bool(torch.isfinite(loss)) and grad_norm > 0 and not torch.cuda.is_initialized(),
          {"loss": float(loss), "n_supervised_positions": int(labels14.numel()),
           "grad_abs_sum": grad_norm, "optimizer_steps": 0, "device": "cpu",
           "mock": True,
           "real_projector_params_llm4096": A.projector_param_count(4096)},
          label="MOCK_FORWARD")

    # --------------------------------------------------- B8 no future leak
    st = state_ep.copy()
    t = 40
    base = RC.proprio42(st[t], st[t - 1], st[t - 2], contract)
    noisy = st.copy()
    rng = np.random.RandomState(0)
    noisy[t + 1:] = (rng.randn(*noisy[t + 1:].shape) * 5.0).astype(np.float32)
    noisy_feat = RC.proprio42(noisy[t], noisy[t - 1], noisy[t - 2], contract)
    check("B8_features_ignore_future", float(np.abs(base - noisy_feat).max()) == 0.0,
          {"max_abs_delta": float(np.abs(base - noisy_feat).max())}, label="VERIFIED_CPU")

    # --------------------------------------------------------- B9 fail closed
    fails = {}
    try:
        RC.load_frozen_contract(os.path.join(REPO, "evidence", "r8_vla_42d_smoke_20261020",
                                            "frozen_r8_contract.json"))
        fails["legacy_80_episode_contract"] = False
    except (ValueError, FileNotFoundError):
        fails["legacy_80_episode_contract"] = True
    for key, fn in (
            ("projector_dim_14", lambda: RC.assert_projector_dim(14)),
            ("frame_t_1", lambda: RC.check_frame_index(1)),
            ("absolute_target_with_42d", lambda: RC.assert_target_mode(AT.ABSOLUTE))):
        try:
            fn()
            fails[key] = False
        except ValueError:
            fails[key] = True
    check("B9_fail_closed", all(fails.values()), fails, label="VERIFIED_CPU")

    # ------------------------------------------------------- B10 PENDING_GPU scope
    pending = {"obs": cuda_lines(TP.obs), "build": cuda_lines(TP.build),
               "evaluate": cuda_lines(TP.evaluate)[:3]}
    check("B10_cuda_only_scope_declared",
          bool(pending["obs"]) and bool(pending["build"]),
          {"cuda_only_lines": {k: len(v) for k, v in pending.items()},
           "sample": pending["obs"][:2],
           "statement": "real 7B forward/backward/loss on GPU is NOT run and NOT claimed"},
          label="PENDING_GPU")

    # ------------------------------------------ B11 entrypoint constants audit (risk)
    import prismatic.vla.constants as PC
    eval_src = inspect.getsource(TP.evaluate)
    uses_action_dim = [ln.strip() for ln in eval_src.splitlines() if "ACTION_DIM" in ln]
    r9_src = open(os.path.join(REPO, "training", "r9_train_pushT_svacc.py")).read()
    consts = {"platform_detected_by": "sys.argv substring (no aloha/libero token -> LIBERO)",
              "ACTION_DIM": int(PC.ACTION_DIM), "NUM_ACTIONS_CHUNK": int(PC.NUM_ACTIONS_CHUNK),
              "PROPRIO_DIM": int(PC.PROPRIO_DIM),
              "codec_expects_action_dim": int(RC.BASE_DIM),
              "evaluate_uses_action_dim": uses_action_dim,
              "r9_entrypoint_references_ACTION_DIM": "ACTION_DIM" in r9_src}
    mismatch = int(PC.ACTION_DIM) != int(RC.BASE_DIM)
    impact = ("untouched entrypoint decodes only ACTION_DIM tokens in evaluate() in this env; "
              "the R9 entrypoint never uses that constant and decodes 14")
    check("B11_entrypoint_constants_audit",
          bool(uses_action_dim) and not consts["r9_entrypoint_references_ACTION_DIM"],
          dict(consts, mismatch=mismatch, impact=impact, status="HYPOTHESIS_OPEN_RISK"),
          label="HYPOTHESIS")

    trace["checks"] = RESULTS
    trace["pass"] = all(r["pass"] for r in RESULTS)
    trace["n_pass"] = sum(1 for r in RESULTS if r["pass"])
    trace["n_total"] = len(RESULTS)
    trace["integration_contract"] = RC.integration_contract(
        contract, meta, args.contract, bounds, os.environ["PUSHT_SPLIT"])
    trace["standard_entrypoint_gaps"] = RC.standard_entrypoint_gaps()
    trace["torch"] = torch.__version__
    trace["cuda_initialized"] = bool(torch.cuda.is_initialized())
    path = os.path.join(args.out, "r9_trainer_trace.json")
    with open(path, "w") as fh:
        json.dump(trace, fh, indent=1)
    print("\n%d/%d checks passed -> %s" % (trace["n_pass"], trace["n_total"],
                                            "PASS" if trace["pass"] else "FAIL"))
    print("trace ->", path)
    return 0 if trace["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())