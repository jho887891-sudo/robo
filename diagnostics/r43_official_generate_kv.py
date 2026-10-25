#!/usr/bin/env python3
"""R43: three-path comparison of the R42 14-token decoder on the real OpenVLA-7B.

Paths
-----
A  full recompute            : 14 multimodal forwards per control step (the deployed default).
B  full recompute + vision   : the same 14 forwards, three cameras encoded once per step.
C  official generate + KV    : ONE multimodal prefill through model.generate(use_cache=True),
                                then 13 incremental decode steps.

Every path uses the same weights, prompt, preprocessed tensor, 42D feature and the same
decode_with_metadata/SAFE/rate contract.  The script never trains, never starts a simulator,
never writes a checkpoint or HDF5 file, never sends a robot command.  All inputs are
deterministic synthetic RGB + 14D states, and 20 offline requests are NOT a closed-loop or
real-time guarantee.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_ROOT = os.path.join(ROOT, "code", "xpolicylab", "policy")
POLICY_DIR = os.path.join(POLICY_ROOT, "OpenVLA_R42")
PURE_DIR = os.path.join(POLICY_ROOT, "OpenVLA_Pure")
for _p in (POLICY_ROOT, POLICY_DIR, PURE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ARX_MARKER = "arx_x5_push_T"


def _force_arx_marker():
    joined = " ".join(sys.argv).lower()
    if "arx" not in joined and "goai" not in joined:
        sys.argv.append(_ARX_MARKER)
        return True
    return False


_TOP_MARKER = _force_arx_marker()
import r42_adapter as R              # noqa: E402
import OpenVLA_R42.model as M        # noqa: E402
import r42_official_generate as OG   # noqa: E402
# NOTE: the ARX marker stays in sys.argv for the whole process. prismatic.vla.constants is
# imported lazily inside _load_prismatic(), and it selects ACTION_DIM=14 from sys.argv at that
# moment. Removing the marker early makes it fall back to the LIBERO constants (ACTION_DIM=7)
# and assert_runtime_contract() then refuses to serve - which is exactly what happened in the
# first R43 debug run. argparse therefore uses parse_known_args().

CAMERAS = tuple(R.CAMERA_ORDER)
SCHEMA = "r43-official-generate-kv/v1"
WALL_BUDGET_S = 30 * 60


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path, chunk=4 * 1024 * 1024):
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
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def pct(values, q):
    a = np.asarray(values, dtype=float)
    return float(np.percentile(a, q)) if a.size else None


def synthetic_rgb(case_index, level):
    """Deterministic 480x640x3 uint8 frame; changes with the case index."""
    h, w = 480, 640
    img = np.empty((h, w, 3), dtype=np.uint8)
    xs = (np.arange(w, dtype=np.uint16) + case_index * 7) % 256
    ys = (np.arange(h, dtype=np.uint16)[:, None] + level) % 256
    img[:, :, 0] = np.broadcast_to(xs[None, :], (h, w)).astype(np.uint8)
    img[:, :, 1] = np.broadcast_to(ys, (h, w)).astype(np.uint8)
    img[:, :, 2] = np.uint8((case_index * 13 + 40) % 256)
    return np.ascontiguousarray(img)


def build_cases(n_cases, amin, amax, seed=20261025):
    rng = np.random.RandomState(seed)
    lo = np.asarray(amin, np.float32)
    hi = np.asarray(amax, np.float32)
    cases = []
    for i in range(n_cases):
        u = rng.rand(14).astype(np.float32)
        # 5% inside the physical box on both sides: nothing is clipped by construction
        state = (lo + (hi - lo) * (0.05 + 0.90 * u)).astype(np.float32)
        images = {cam: synthetic_rgb(i, level)
                  for cam, level in zip(CAMERAS, (96, 127, 160))}
        cases.append({"index": i, "state": state, "vision": images})
    return cases


class TokenTimer:
    """Per-token forward latency for paths A/B (wraps backend.next_token_logits)."""

    def __init__(self, backend, torch):
        self.backend = backend
        self.torch = torch
        self.times = []
        self.windows = []
        self._orig = backend.next_token_logits

        def wrap(*a, **k):
            t0 = time.perf_counter()
            out = self._orig(*a, **k)
            self.torch.cuda.synchronize()
            self.times.append((time.perf_counter() - t0) * 1e3)
            if out is not None:
                w = np.asarray(out).reshape(-1)
                self.windows.append(
                    w[OG.LEGAL_WINDOW[0]:OG.LEGAL_WINDOW[1] + 1].astype(np.float32))
            return out

        backend.next_token_logits = wrap

    def reset(self):
        self.times = []
        self.windows = []

    def pop(self):
        v = list(self.times)
        self.times = []
        return v

    def pop_windows(self):
        v = list(self.windows)
        self.windows = []
        return v

    def close(self):
        self.backend.next_token_logits = self._orig


class VisionCounter:
    """Counts real VisionBackbone encodes (must be installed BEFORE install_vision_cache)."""

    def __init__(self, core):
        self.vb = core.vision_backbone
        self.n = 0
        self._orig = self.vb.forward

        def wrap(*a, **k):
            self.n += 1
            return self._orig(*a, **k)

        self.vb.forward = wrap

    def reset(self):
        self.n = 0

    def close(self):
        self.vb.forward = self._orig


def run_full(server, case, timer, vision, torch, path):
    server.reset_all()
    timer.reset()
    vision.reset()
    t0 = time.perf_counter()
    out = server.act(case["vision"], case["state"], env_idx=0,
                     prev_cmd=case["state"].copy(), dt=0.04)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    return {
        "ids": [int(x) for x in out["ids"]],
        "bins": [int(x) for x in out["bins"]],
        "action": np.asarray(out["action"], np.float64).tolist(),
        "residual": np.asarray(out["residual"], np.float64).tolist(),
        "info": dict(out.get("info") or {}),
        "latency_ms": dict(out["latency_ms"]),
        "wall_ms": wall,
        "token_ms": timer.pop(),
        "window_logits": timer.pop_windows(),
        "vision_calls": int(vision.n),
        "path": path,
    }


def run_official(server, adapter, case, torch):
    server.reset_all()
    feat = server._history.observe(0, case["state"])
    pv = server.preprocess(case["vision"])
    res = adapter.generate(server.prompt_ids(), pv, feat)
    torch.cuda.synchronize()
    v = server._history.velocity(0)
    action, residual, info = R.decode_with_metadata(
        server.metadata, np.asarray(res["bins"]), case["state"], v,
        prev_cmd=case["state"].copy(), dt=0.04)
    token_ms = [float(res["info"]["prefill_ms"])] + [float(x) for x in res["info"]["decode_ms"]]
    return {
        "ids": [int(x) for x in res["ids"]],
        "bins": [int(x) for x in res["bins"]],
        "action": np.asarray(action, np.float64).tolist(),
        "residual": np.asarray(residual, np.float64).tolist(),
        "info": dict(info),
        "latency_ms": {"forward_total": float(res["info"]["wall_ms"])},
        "wall_ms": float(res["info"]["wall_ms"]),
        "token_ms": token_ms,
        "vision_calls": int(res["info"]["vision_calls"]),
        "path": "C",
        "gen_info": res["info"],
        "window_logits": res["window_logits"],
        "free_argmax_ids": res["free_argmax_ids"],
        "margins": res["margins"],
        "prefill_ms": float(res["info"]["prefill_ms"]),
    }


def collect_environment(torch, backend_files):
    env = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "files": {},
    }
    try:
        import transformers
        env["transformers"] = transformers.__version__
        env["transformers_file"] = transformers.__file__
    except Exception as exc:                                  # pragma: no cover
        env["transformers"] = "missing: %s" % exc
    try:
        import peft
        env["peft"] = getattr(peft, "__version__", "unknown")
    except Exception as exc:                                  # pragma: no cover
        env["peft"] = "missing: %s" % exc
    try:
        import prismatic
        env["prismatic_file"] = prismatic.__file__
        import prismatic.vla.constants as C
        env["prismatic_constants"] = {
            "ACTION_DIM": int(getattr(C, "ACTION_DIM")),
            "ACTION_TOKEN_BEGIN_IDX": int(getattr(C, "ACTION_TOKEN_BEGIN_IDX")),
            "STOP_INDEX": int(getattr(C, "STOP_INDEX")),
        }
    except Exception as exc:                                  # pragma: no cover
        env["prismatic_file"] = "missing: %s" % exc
    for name, path in backend_files.items():
        env["files"][name] = {"path": path, "sha256": sha256_file(path) if os.path.isfile(path) else None}
    return env


def compare_sequences(results_a, results_c, label):
    """Token-level equality with the first divergence described in context."""
    total = 0
    equal = 0
    first = None
    per_case = []
    for (ca, cc) in zip(results_a, results_c):
        toks_a, toks_c = ca["ids"], cc["ids"]
        same = toks_a == toks_c
        n = len(toks_a)
        total += n
        equal += sum(1 for x, y in zip(toks_a, toks_c) if x == y)
        if not same and first is None:
            k = next(i for i, (x, y) in enumerate(zip(toks_a, toks_c)) if x != y)
            wa = ca["window_logits"][k]
            wc = cc["window_logits"][k]
            order = np.argsort(wa)[-2:][::-1]
            first = {
                "case_index": ca["case_index"],
                "token_index": k,
                "prefix_tokens_A": toks_a[:k],
                "token_A": toks_a[k],
                "token_C": toks_c[k],
                "margin_A": float(wa[order[0]] - wa[order[1]]),
                "margin_C": float(np.sort(wc)[-1] - np.sort(wc)[-2]),
                "max_abs_logit_delta_window": float(np.max(np.abs(wa - wc))),
                "mean_abs_logit_delta_window": float(np.mean(np.abs(wa - wc))),
                "argmax_A_bin": int(order[0]),
                "argmax_C_bin": int(np.argmax(wc)),
                "top1_A": float(wa[order[0]]),
                "top2_A": float(wa[order[1]]),
                "action_A": ca["action"],
                "action_C": cc["action"],
            }
        per_case.append({
            "case_index": ca["case_index"],
            "identical": bool(same),
            "n_differing_tokens": int(sum(1 for x, y in zip(toks_a, toks_c) if x != y)),
            "max_abs_action_delta": float(np.max(np.abs(np.asarray(ca["action"])
                                                        - np.asarray(cc["action"])))),
        })
    return {
        "label": label,
        "requests": len(results_a),
        "tokens_total": total,
        "tokens_equal": equal,
        "token_agreement_rate": (equal / total) if total else None,
        "cases_identical": sum(1 for p in per_case if p["identical"]),
        "all_identical": bool(equal == total),
        "first_divergence": first,
        "per_case": per_case,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deploy-yml", default=os.path.join(POLICY_DIR, "deploy.yml"))
    ap.add_argument("--out", required=True, help="evidence directory OUTSIDE the repository")
    ap.add_argument("--min-free-gb", type=float, default=24.0)
    ap.add_argument("--cases", type=int, default=20)
    args, _unknown = ap.parse_known_args()

    out = os.path.abspath(os.path.expanduser(args.out))
    if os.path.exists(out):
        print("BLOCKED: refusing to reuse an existing directory: " + out, file=sys.stderr)
        return 2
    if os.path.commonpath([ROOT, out]) == ROOT:
        print("BLOCKED: --out must be outside the repository", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)

    report = {
        "schema_version": SCHEMA,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "source": {
            "repo_root": ROOT,
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_status": _git("status", "--short"),
        },
        "scope": {
            "requests_per_path": args.cases,
            "paths": ["A_full_recompute", "B_full_recompute_vision_cache",
                      "C_official_generate_kv"],
            "training_or_optimizer_steps": 0,
            "simulator_started": False,
            "robot_connected_or_commanded": False,
            "checkpoint_or_hdf5_mutation": False,
            "real_observations_used": False,
            "synthetic_rgb_state_used": True,
        },
        "checks": [],
        "artifacts": {},
    }
    exit_code = 1
    torch = None
    try:
        import yaml
        with open(args.deploy_yml, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if not isinstance(config, dict):
            raise _Blocked("deploy YAML did not decode to a mapping")
        report["deployment_config"] = {
            "policy_name": config.get("policy_name"),
            "checkpoint_dir": config.get("checkpoint_dir"),
            "base_model_path": config.get("base_model_path"),
            "control_hz": config.get("control_hz"),
            "device": config.get("device"),
        }
        if str(config.get("device", "")).lower() != "cuda":
            raise _Blocked("frozen deploy config does not select CUDA")
        if abs(float(config.get("control_hz", 0)) - 25.0) > 1e-9:
            raise _Blocked("expected control_hz=25, got %r" % config.get("control_hz"))

        import torch as _torch
        torch = _torch
        if not torch.cuda.is_available():
            raise _Blocked("CUDA unavailable")
        free_b, total_b = torch.cuda.mem_get_info()
        report["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "free_gb": round(free_b / 1e9, 3),
                         "total_gb": round(total_b / 1e9, 3),
                         "min_free_gb": args.min_free_gb}
        _record(report, "free_vram_gate", free_b / 1e9 >= args.min_free_gb, report["gpu"])
        if free_b / 1e9 < args.min_free_gb:
            raise _Blocked("free VRAM below the %.1f GB gate; nothing was loaded" % args.min_free_gb)

        t_load = time.perf_counter()
        model = M.Model(config)
        model.warmup()
        report["model_load_s"] = time.perf_counter() - t_load
        _deadline[0] = time.time() + WALL_BUDGET_S
        report["measurement_budget_s"] = WALL_BUDGET_S
        server = model._server
        backend = server.backend
        core = backend.core
        torch.cuda.synchronize()
        _record(report, "model_loaded", core is not None,
                {"backend": type(backend).__name__, "load_s": round(report["model_load_s"], 2)})
        _record(report, "pure_autoreg_installed",
                bool(getattr(core, "_pure_autoreg_input_mode", False)))
        try:
            n_img = int(core.vision_backbone.get_num_images_in_input())
        except Exception:
            n_img = None
        _record(report, "three_camera_vision", n_img == 3, {"num_images": n_img})

        import transformers as _tf
        backend_files = {
            "modeling_prismatic.py": os.path.join(ROOT, "code", "openvla_dynamic_modules",
                                                  "modeling_prismatic.py"),
            "transformers_generation_utils.py": os.path.join(
                os.path.dirname(_tf.__file__), "generation", "utils.py"),
            "r42_deploy_server.py": os.path.join(POLICY_DIR, "r42_deploy_server.py"),
            "r42_adapter.py": os.path.join(POLICY_DIR, "r42_adapter.py"),
            "r42_official_generate.py": os.path.join(POLICY_DIR, "r42_official_generate.py"),
            "pure_autoreg_mode.py": os.path.join(PURE_DIR, "pure_autoreg_mode.py"),
        }
        report["environment"] = collect_environment(torch, backend_files)
        report["prismatic_runtime_contract"] = R.assert_runtime_contract()

        meta = server.metadata
        lo = np.asarray(meta["safety_limits"]["amin"], np.float32)
        hi = np.asarray(meta["safety_limits"]["amax"], np.float32)
        rate = np.asarray(meta["safety_limits"]["max_rate"], np.float32)
        _record(report, "rate_contract_14d",
                lo.shape == hi.shape == rate.shape == (14,), {"max_rate14": rate.tolist()})

        # ---- weight identity: the three paths must run on byte-identical weights
        base_dir = config.get("base_model_path")
        ckpt_dir = config.get("checkpoint_dir")
        base_rows = hash_dir(base_dir)
        ckpt_rows = hash_dir(ckpt_dir)
        save_json(os.path.join(out, "base_model_weight_hashes.json"),
                  {"path": base_dir, "files": base_rows,
                   "total_bytes": sum(r["size_bytes"] for r in base_rows)})
        save_json(os.path.join(out, "checkpoint_hashes.json"),
                  {"path": ckpt_dir, "files": ckpt_rows,
                   "lora_dir": getattr(model, "lora_dir", None),
                   "projector_path": getattr(model, "projector_path", None),
                   "total_bytes": sum(r["size_bytes"] for r in ckpt_rows)})
        report["weight_identity"] = {
            "base_model_files": len(base_rows), "base_model_bytes": sum(r["size_bytes"] for r in base_rows),
            "checkpoint_files": len(ckpt_rows), "checkpoint_bytes": sum(r["size_bytes"] for r in ckpt_rows),
            "base_model_sha256_of_manifest": hashlib.sha256(
                json.dumps(base_rows, sort_keys=True).encode()).hexdigest(),
            "checkpoint_sha256_of_manifest": hashlib.sha256(
                json.dumps(ckpt_rows, sort_keys=True).encode()).hexdigest(),
        }
        _record(report, "weights_identified", bool(base_rows) and bool(ckpt_rows),
                report["weight_identity"])

        cases = build_cases(args.cases, lo, hi)
        prompt = server.prompt_ids()
        _record(report, "prompt_frozen", len(prompt) > 0, {"prompt_len": len(prompt)})

        # ---- identity: the object we patch (core) must still apply the LoRA adapter
        c0 = cases[0]
        server.reset_all()
        feat0 = server._history.observe(0, c0["state"])
        pv0 = server.preprocess(c0["vision"])
        logits_peft = backend.next_token_logits(prompt, pv0, feat0)
        ids = torch.tensor([prompt], dtype=torch.long, device=backend.device)
        pr = torch.as_tensor(feat0.reshape(1, -1), dtype=torch.float32, device=backend.device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_core = core(input_ids=ids, attention_mask=torch.ones_like(ids),
                                pixel_values=pv0, proprio=pr,
                                proprio_projector=backend.projector, labels=ids.clone())
        logits_core = out_core.logits[0, -1].float().cpu().numpy()
        delta = float(np.max(np.abs(logits_core - logits_peft)))
        _record(report, "core_path_equals_peft_path", delta <= 1e-5,
                {"max_abs_logit_delta": delta,
                 "note": "core() must apply the same LoRA weights as PeftModel(); otherwise "
                         "generate() on the core would silently drop the adapter"})

        # ---- path A: 14 full forwards
        timer = TokenTimer(backend, torch)
        vision = VisionCounter(core)
        run_full(server, c0, timer, vision, torch, "A")          # warmup
        torch.cuda.reset_peak_memory_stats()
        t_a = time.perf_counter()
        res_a = []
        for c in cases:
            if time.time() > _deadline[0]:
                report["budget_exceeded_in"] = "A"
                break
            r = run_full(server, c, timer, vision, torch, "A")
            r["case_index"] = c["index"]
            res_a.append(r)
        report["path_A_s"] = time.perf_counter() - t_a
        report["path_A_mem_gb"] = {"allocated": torch.cuda.max_memory_allocated() / 1e9,
                                   "reserved": torch.cuda.max_memory_reserved() / 1e9}

        # ---- path B: same 14 forwards, vision encoded once per control step
        _restore_vision(core, vision)
        vision = VisionCounter(core)                            # wraps the ORIGINAL forward
        backend.install_vision_cache()                          # captures the counting wrapper
        run_full(server, c0, timer, vision, torch, "B")         # warmup
        torch.cuda.reset_peak_memory_stats()
        t_b = time.perf_counter()
        res_b = []
        for c in cases:
            if time.time() > _deadline[0]:
                report["budget_exceeded_in"] = "B"
                break
            r = run_full(server, c, timer, vision, torch, "B")
            r["case_index"] = c["index"]
            res_b.append(r)
        report["path_B_s"] = time.perf_counter() - t_b
        report["path_B_mem_gb"] = {"allocated": torch.cuda.max_memory_allocated() / 1e9,
                                   "reserved": torch.cuda.max_memory_reserved() / 1e9}
        vision.close()
        timer.close()

        # ---- path C: official generate + KV cache
        _restore_vision(core, None)
        backend.vision_cache_enabled = False
        backend._vision_cache = None
        adapter = OG.R42OfficialGenerateAdapter(core, backend.projector, device=backend.device)
        adapter.count_vision(core)
        adapter.install()
        run_official(server, adapter, c0, torch)                # warmup
        torch.cuda.reset_peak_memory_stats()
        t_c = time.perf_counter()
        res_c = []
        for c in cases:
            if time.time() > _deadline[0]:
                report["budget_exceeded_in"] = "C"
                break
            r = run_official(server, adapter, c, torch)
            r["case_index"] = c["index"]
            res_c.append(r)
        report["path_C_s"] = time.perf_counter() - t_c
        report["path_C_mem_gb"] = {"allocated": torch.cuda.max_memory_allocated() / 1e9,
                                   "reserved": torch.cuda.max_memory_reserved() / 1e9}
        adapter.uninstall()

        eq_ab = compare_sequences(res_a, res_b, "A_vs_B")
        eq_ac = compare_sequences(res_a, res_c, "A_vs_C")
        report["equivalence"] = {"A_vs_B": _slim(eq_ab), "A_vs_C": _slim(eq_ac)}
        _record(report, "A_vs_B_token_identical", eq_ab["all_identical"],
                {"tokens_equal": eq_ab["tokens_equal"], "tokens_total": eq_ab["tokens_total"]})
        _record(report, "A_vs_C_token_identical", eq_ac["all_identical"],
                {"tokens_equal": eq_ac["tokens_equal"], "tokens_total": eq_ac["tokens_total"]})

        report["performance"] = _performance(res_a, res_b, res_c)
        report["status"] = "PASS" if (eq_ab["all_identical"] and eq_ac["all_identical"]) else "FAIL"
        exit_code = 0 if report["status"] == "PASS" else 3
        _write_artifacts(out, report, res_a, res_b, res_c, eq_ab, eq_ac)
        _finalize(out, report, exit_code)
    except _Blocked as exc:
        report["status"] = "BLOCKED"
        report["blocked_reason"] = str(exc)
        exit_code = 2
    except Exception as exc:                                   # pragma: no cover
        report["status"] = "ERROR"
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
        exit_code = 1
    finally:
        report["finished_at_utc"] = utc_now()
        if not report.get("_finalized"):
            try:
                save_json(os.path.join(out, "report.json"), report)
                with open(os.path.join(out, "exit_code.txt"), "w", encoding="utf-8") as fh:
                    fh.write(str(exit_code) + chr(10))
            except Exception:
                pass
    print("STATUS=%s exit=%d out=%s" % (report.get("status"), exit_code, out), flush=True)
    return exit_code


# ---------------------------------------------------------------------------------- helpers
import subprocess  # noqa: E402

_deadline = [0.0]


class _Blocked(RuntimeError):
    """A precondition was not met; no measurement may be claimed."""


def _git(*args):
    try:
        r = subprocess.run(["git"] + list(args), cwd=ROOT, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return None


def _record(report, name, ok, detail=None):
    row = {"name": name, "result": "PASS" if ok else "FAIL"}
    if detail is not None:
        row["detail"] = detail
    report["checks"].append(row)
    print("[%s] %s %s" % (row["result"], name,
                          json.dumps(detail, sort_keys=True, default=str)[:400]
                          if detail is not None else ""), flush=True)
    return bool(ok)


def _restore_vision(core, counter):
    """Put the ORIGINAL VisionBackbone.forward back, whatever wrapper is installed now."""
    vb = core.vision_backbone
    if counter is not None:
        counter.close()
    orig = getattr(vb, "_r42_vision_orig", None)
    if orig is not None:
        vb.forward = orig
    if getattr(vb, "_r42_vision_cached", False):
        try:
            delattr(vb, "_r42_vision_cached")
        except AttributeError:
            pass
        if hasattr(vb, "_r42_vision_orig"):
            try:
                delattr(vb, "_r42_vision_orig")
            except AttributeError:
                pass


def hash_dir(path, patterns=("*.safetensors", "*.json", "*.pt", "*.bin")):
    """Sha256 of the weight/config files that define this checkpoint's identity."""
    import fnmatch
    rows = []
    if not path or not os.path.isdir(path):
        return rows
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if not any(fnmatch.fnmatch(name, pat) for pat in patterns):
                continue
            full = os.path.join(root, name)
            rows.append({"path": os.path.relpath(full, path),
                         "size_bytes": os.path.getsize(full),
                         "sha256": sha256_file(full)})
    return rows


def _slim(eq):
    first = eq["first_divergence"]
    if first is not None:
        first = dict(first)
        first.pop("action_A", None)
        first.pop("action_C", None)
        first["prefix_tokens_A_len"] = len(first.get("prefix_tokens_A") or [])
    return {"label": eq["label"], "requests": eq["requests"],
            "tokens_total": eq["tokens_total"], "tokens_equal": eq["tokens_equal"],
            "token_agreement_rate": eq["token_agreement_rate"],
            "cases_identical": eq["cases_identical"], "all_identical": eq["all_identical"],
            "first_divergence": first}


def _stats(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max()), "p50": pct(a, 50), "p95": pct(a, 95), "p99": pct(a, 99)}


def _path_perf(results, label):
    req = [r["wall_ms"] for r in results]
    toks = [t for r in results for t in r["token_ms"]]
    out = {"label": label, "requests": len(results),
           "request_ms": _stats(req), "token_ms": _stats(toks),
           "vision_calls_per_request": _stats([r["vision_calls"] for r in results])}
    if results and results[0]["path"] == "C":
        out["prefill_ms"] = _stats([r["prefill_ms"] for r in results])
        out["decode_ms"] = _stats([t for r in results for t in r["token_ms"][1:]])
        out["prefill_calls_per_request"] = _stats(
            [r["gen_info"]["prefill_calls"] for r in results])
        out["decode_calls_per_request"] = _stats(
            [r["gen_info"]["decode_calls"] for r in results])
    return out


def _performance(res_a, res_b, res_c):
    pa, pb, pc = (_path_perf(res_a, "A_full_recompute"),
                  _path_perf(res_b, "B_full_recompute_vision_cache"),
                  _path_perf(res_c, "C_official_generate_kv"))
    perf = {"A": pa, "B": pb, "C": pc, "period_ms": 40.0}
    if pa["request_ms"] and pb["request_ms"] and pc["request_ms"]:
        a50 = pa["request_ms"]["p50"]
        perf["speedup_vs_A"] = {
            "B": a50 / pb["request_ms"]["p50"] if pb["request_ms"]["p50"] else None,
            "C": a50 / pc["request_ms"]["p50"] if pc["request_ms"]["p50"] else None,
        }
        perf["realtime_multiple_of_40ms"] = {
            "A": a50 / 40.0,
            "B": pb["request_ms"]["p50"] / 40.0,
            "C": pc["request_ms"]["p50"] / 40.0,
        }
    return perf


def _write_artifacts(out, report, res_a, res_b, res_c, eq_ab, eq_ac):
    def path(name):
        return os.path.join(out, name)

    for tag, res in (("A", res_a), ("B", res_b), ("C", res_c)):
        if res:
            np.save(path("logits_window_%s.npy" % tag),
                    np.stack([r["window_logits"] for r in res]))

    with open(path("latency_by_token.jsonl"), "w", encoding="utf-8") as fh:
        for tag, res in (("A", res_a), ("B", res_b), ("C", res_c)):
            for r in res:
                for k, ms in enumerate(r["token_ms"]):
                    fh.write(json.dumps({"path": tag, "case_index": r["case_index"],
                                         "token_index": k, "ms": float(ms)},
                                        sort_keys=True) + "\n")

    seq = {}
    for tag, res in (("A", res_a), ("B", res_b), ("C", res_c)):
        seq[tag] = [{k: r[k] for k in ("case_index", "ids", "bins", "action", "residual",
                                       "info", "vision_calls", "wall_ms")} for r in res]
    env = report.get("environment") or {}
    with open(path("environment.txt"), "w", encoding="utf-8") as fh:
        for key in sorted(env):
            if key != "files":
                fh.write("%s: %s" % (key, env[key]) + chr(10))
        for key, row in sorted((env.get("files") or {}).items()):
            fh.write("file %s sha256=%s path=%s" % (key, row.get("sha256"), row.get("path"))
                     + chr(10))
    save_json(path("sequences.json"), seq)
    save_json(path("equivalence_report.json"),
              {"A_vs_B": eq_ab, "A_vs_C": eq_ac})
    report["artifacts"]["files"] = sorted(os.listdir(out))


def _finalize(out, report, exit_code):
    save_json(os.path.join(out, "report.json"), report)
    with open(os.path.join(out, "exit_code.txt"), "w", encoding="utf-8") as fh:
        fh.write("%d\n" % exit_code)
    names = sorted(n for n in os.listdir(out) if n != "SHA256SUMS")
    lines = ["%s  %s" % (sha256_file(os.path.join(out, n)), n) for n in names]
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    report["_finalized"] = True


if __name__ == "__main__":
    sys.exit(main())
