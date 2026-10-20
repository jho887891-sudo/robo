#!/usr/bin/env python3
"""R10 regression: full-logit sidecar + KV cache preflight in diagnostics/r8_smoke.py."""
import json
import os
import re
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "diagnostics"))
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r8_smoke as S  # noqa: E402

RESULTS = []


def ok(name, cond, detail=None):
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail or {}})
    print("[%s] %-42s %s" % ("PASS" if cond else "FAIL", name,
                             json.dumps(detail or {}, default=str)[:180]))
    return bool(cond)


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True


class StubVLA:
    """Minimal stand-in with the attributes the preflight touches."""

    def __init__(self, gc=False):
        self.is_gradient_checkpointing = gc
        self.config = type("C", (), {"use_cache": False})()
        self.disabled_calls = 0

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False
        self.disabled_calls += 1


def main():
    ok("has_sidecar_helper", callable(getattr(S, "save_kv_logits_sidecar", None)))
    ok("has_preflight_helper", callable(getattr(S, "kv_preflight", None)))

    gib = float(2 ** 30)
    ok("preflight_rejects_low_vram",
       raises(lambda: S.kv_preflight(None, int(23.3 * gib), 24.0), RuntimeError),
       {"free_gib": 23.3, "min": 24.0})
    ok("preflight_accepts_enough_vram",
       S.kv_preflight(None, int(25.0 * gib), 24.0)["free_gib"] > 24.0)

    stub = StubVLA(gc=True)
    info = S.kv_preflight(stub, int(25.0 * gib), 24.0)
    ok("preflight_disables_gradient_checkpointing",
       stub.disabled_calls == 1 and info["gradient_checkpointing_after"] is False
       and info["gradient_checkpointing_before"] is True, info)
    ok("preflight_forces_use_cache", bool(stub.config.use_cache) is True)

    with tempfile.TemporaryDirectory() as td:
        kv = np.random.RandomState(0).randn(14, 256).astype(np.float32)
        fl = kv + 0.01
        meta = S.save_kv_logits_sidecar(td, kv, fl, list(range(14)), list(range(14)))
        path = meta["path"]
        data = open(path, "rb").read()
        ok("sidecar_written_with_hash",
           os.path.isfile(path) and meta["sha256"] == __import__("hashlib").sha256(data).hexdigest()
           and meta["shape"] == [14, 256] and meta["dtype"] == "float32",
           {"bytes": meta["bytes"], "shape": meta["shape"]})
        z = np.load(path)
        ok("sidecar_roundtrip_exact",
           np.array_equal(z["logits_kv"], kv) and np.array_equal(z["logits_full"], fl)
           and z["logits_kv"].shape == (14, 256),
           {"keys": sorted(z.keys())})
        ok("sidecar_rejects_wrong_width",
           raises(lambda: S.save_kv_logits_sidecar(td, np.zeros((14, 255)), np.zeros((14, 255)),
                                                  [0] * 14, [0] * 14), ValueError))
        ok("sidecar_rejects_token_mismatch",
           raises(lambda: S.save_kv_logits_sidecar(td, kv, fl, [0] * 13, [0] * 14), ValueError))
        bad = kv.copy()
        bad[0, 0] = np.nan
        ok("sidecar_rejects_nonfinite",
           raises(lambda: S.save_kv_logits_sidecar(td, bad, fl, list(range(14)), list(range(14))),
                  ValueError))

    src = open(os.path.join(REPO, "diagnostics", "r8_smoke.py")).read()
    ok("kv_diagnose_uses_preflight_and_sidecar",
       "kv_preflight(" in src and "save_kv_logits_sidecar(" in src
       and "gradient_checkpointing_disable" in src)
    kv_block = src[src.index("def kv_diagnose("):src.index("\ndef main():")]
    ok("kv_diagnose_no_gradient_checkpointing_enable",
       "gradient_checkpointing_enable()" not in kv_block,
       {"scoped_to": "kv_diagnose only (the main smoke mode still uses it for backward)"})
    ok("kv_diagnose_rejects_missing_cache",
       "no KV cache returned" in src and "refusing to fake a comparison" in src)
    ok("kv_diagnose_records_diag_executed", "DIAG_EXECUTED" in src and "KV_EQUIVALENT" in src)
    ok("kv_window_constants_unchanged", S.W0 == 31744 and S.BINS == 256)

    n = sum(1 for r in RESULTS if r["pass"])
    print("\n%d/%d checks passed" % (n, len(RESULTS)))
    out = os.path.join(REPO, "evidence", "r10", "test_r10_kv_probe.json")
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w") as fh:
            json.dump({"checks": RESULTS, "n_pass": n, "n_total": len(RESULTS)}, fh, indent=1)
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
