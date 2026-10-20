#!/usr/bin/env python3
"""Regression for diagnostics/r9_kv_same_prefix.py (R9 Gate C CPU contract)."""
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "diagnostics"))

import r9_kv_same_prefix as K  # noqa: E402

RESULTS = []


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-42s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:200]))
    return bool(cond)


def fake_runner(case):
    """Deterministic stand-in for cached vs full branches (no model involved)."""
    state = {"fed": []}

    def full_vec():
        v = np.zeros(8)
        v[0] = 0.1
        v[1] = 0.09
        return v

    def prefill(prefix):
        state["fed"] = []
        return full_vec()

    def cached_step(token):
        state["fed"].append(int(token))
        if case == "clean":
            return full_vec()
        if case == "numeric":
            v = full_vec()
            v[1] = 0.105          # flips argmax with a 0.015 logit delta
            return v
        v = np.zeros(8)
        v[0] = 0.1
        v[1] = 6.0                # flips argmax with a ~5.9 logit delta
        return v

    def full(seq):
        return full_vec()

    return prefill, cached_step, full, state


def main():
    window = (0, 8)
    pre, step, full, state = fake_runner("clean")
    steps = K.same_prefix_probe(pre, step, full, [1, 2, 3], n_steps=5, window=window)
    s = K.summarize(steps)
    ok("clean_run_is_equivalent", s["verdict"] == "SAME_PREFIX_EQUIVALENT",
       {"verdict": s["verdict"], "n_match": s["n_match"]})
    ok("exit_code_zero_on_match", K.exit_code(s) == 0)
    ok("teacher_forcing_uses_full_tokens",
       state["fed"] == [st["token_full"] for st in steps[:-1]],
       {"fed": state["fed"]})
    ok("prefix_len_grows_by_one",
       [st["prefix_len"] for st in steps] == [3, 4, 5, 6, 7])

    pre, step, full, _ = fake_runner("numeric")
    s2 = K.summarize(K.same_prefix_probe(pre, step, full, [1, 2, 3], n_steps=4, window=window))
    ok("small_delta_classed_numeric_plausible",
       s2["mismatch_class"] == "numeric_plausible" and s2["verdict"].startswith("OPEN_NUMERIC"),
       {"class": s2["mismatch_class"], "delta": s2["max_abs_dlogit_at_mismatch"]})
    ok("exit_code_nonzero_on_mismatch", K.exit_code(s2) == 6)

    pre, step, full, _ = fake_runner("structural")
    s3 = K.summarize(K.same_prefix_probe(pre, step, full, [1, 2, 3], n_steps=4, window=window))
    ok("huge_delta_classed_structural",
       s3["mismatch_class"] == "structural_suspect"
       and s3["verdict"].startswith("OPEN_STRUCTURAL"),
       {"class": s3["mismatch_class"], "delta": s3["max_abs_dlogit_at_mismatch"]})

    ok("window_slice", np.array_equal(K._window(np.arange(16.0), (4, 3)),
                                      np.array([4.0, 5.0, 6.0])))

    # Authorization token alone must never be mistaken for successful GPU execution.
    import subprocess
    wrapper = os.path.join(REPO, "diagnostics", "r9_kv_same_prefix.py")
    env_yes = dict(os.environ, R9_KV_GPU_AUTHORIZED="1")
    p_yes = subprocess.run([sys.executable, wrapper, "--gpu-diagnose"],
                           env=env_yes, capture_output=True, text=True)
    ok("authorized_wrapper_does_not_fake_gpu_pass",
       p_yes.returncode == 8 and "no GPU probe was run" in p_yes.stderr,
       {"exit": p_yes.returncode})
    env_no = dict(os.environ)
    env_no.pop("R9_KV_GPU_AUTHORIZED", None)
    p_no = subprocess.run([sys.executable, wrapper, "--gpu-diagnose"],
                          env=env_no, capture_output=True, text=True)
    ok("unauthorized_wrapper_still_fails_closed",
       p_no.returncode == 7, {"exit": p_no.returncode})

    audit = K.static_audit(os.path.join(REPO, "diagnostics", "r8_smoke.py"))
    ok("static_audit_real_probe_passes", audit["pass"],
       {"missing": [k for k, v in audit["required"].items() if not v["found"]],
        "forbidden_found": [k for k, v in audit["forbidden"].items() if v["found"]]})

    bad_src = ("def kv():\n"
               "    cached = None\n"
               "    o = vla(input_ids=torch.tensor([[W0 + tk]]), past_key_values=cached)\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(bad_src)
        bad_path = fh.name
    try:
        bad = K.static_audit(bad_path)
        ok("static_audit_detects_old_bug", bad["pass"] is False
           and bad["forbidden"]["advance_cache_with_cached_token"]["found"] is True)
    finally:
        os.remove(bad_path)

    n = sum(1 for r in RESULTS if r["pass"])
    print("\n%d/%d checks passed" % (n, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r9_kv_cpu_20261020", "test_r9_kv_same_prefix.json")
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as fh:
            json.dump({"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS)}, fh, indent=1)
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
