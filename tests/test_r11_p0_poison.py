#!/usr/bin/env python3
"""R11 P0-2 (RED first): prismatic ACTION_DIM=7 / NUM_ACTIONS_CHUNK=8 / PROPRIO_DIM=8 must not
leak into the 14D R42 path.

prismatic.vla.constants picks its values from sys.argv (LIBERO 7D by default, ARX 14D only when
"arx"/"goai" is present).  Any 14-token autoregressive loop or 42D projector that relies on those
globals is silently wrong.  The R42 path must (a) assert the runtime really carries ACTION_DIM=14
and the 31743/2 codec constants before touching the model, and (b) never use the chunked/prismatic
dimension defaults itself.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402

def _prismatic_root():
    """Pin the runtime the R42 path is launched against.

    Other test modules set PUSHT_OFT at import time, which would otherwise leak a *different*
    prismatic build (with a different constants table) into these subprocesses.  The published
    Push-T runtime wins; PUSHT_OFT is only a fallback.
    """
    for cand in ("/home/dgut/pusht_runtime/openvla_oft_src", os.environ.get("PUSHT_OFT")):
        if cand and os.path.isdir(os.path.join(cand, "prismatic")):
            return cand
    return None


OFT = _prismatic_root()
HAS_PRISMATIC = OFT is not None

# Files on the R42 path that must never use the prismatic dimension globals.
R42_SOURCES = [
    os.path.join("code", "xpolicylab", "policy", "OpenVLA_R42", "r42_adapter.py"),
    os.path.join("code", "xpolicylab", "policy", "OpenVLA_R42", "r42_deploy_server.py"),
    os.path.join("training", "r42_train_contract.py"),
    os.path.join("training", "r11_train_pushT_r42.py"),
]
# Static patterns that would mean a legacy 7D/8D/chunked default drives the 14D path.
DANGEROUS = [
    "range(ACTION_DIM)",
    "range(NUM_ACTIONS_CHUNK)",
    "ACTION_DIM *",
    "* NUM_ACTIONS_CHUNK",
    "proprio_dim=PROPRIO_DIM",
    "proprio_dim=7",
    "proprio_dim=8",
]


def _run(code):
    # the child sets sys.argv itself before importing prismatic, so no argv is passed here
    root = _prismatic_root()
    assert root, "no prismatic runtime found"
    env = dict(os.environ)
    env.pop("PUSHT_OFT", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [root, os.path.join(root, "vla-scripts"),
         os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"),
         os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure")])
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.mark.skipif(not HAS_PRISMATIC, reason="prismatic runtime not installed here")
def test_libero_defaults_are_rejected():
    """With no arx marker in argv prismatic reports 7D; the R42 guard must fail closed."""
    code = (
        "import sys, json\n"
        "sys.argv = ['probe']\n"
        "import r42_adapter as R\n"
        "try:\n"
        "    R.assert_runtime_contract()\n"
        "except R.R42Error as e:\n"
        "    print('REJECTED', str(e)[:120]); raise SystemExit(3)\n"
        "print('ACCEPTED'); raise SystemExit(0)\n"
    )
    res = _run(code)
    assert res.returncode == 3, "expected fail-closed rejection, got rc=%d\n%s\n%s" % (
        res.returncode, res.stdout[-2000:], res.stderr[-2000:])


@pytest.mark.skipif(not HAS_PRISMATIC, reason="prismatic runtime not installed here")
def test_arx_14d_runtime_is_accepted_and_recorded():
    code = (
        "import sys, json\n"
        "sys.argv = ['run', 'arx_x5_push_T']\n"
        "import r42_adapter as R\n"
        "import prismatic, os\n"
        "obs = R.assert_runtime_contract()\n"
        "obs['prismatic_file'] = os.path.dirname(prismatic.__file__)\n"
        "print(json.dumps(obs, sort_keys=True))\n"
    )
    res = _run(code)
    assert res.returncode == 0, "arx runtime rejected: rc=%d\n%s\n%s" % (
        res.returncode, res.stdout[-2000:], res.stderr[-2000:])
    payload = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert payload, "no JSON evidence on stdout:\n%s" % res.stdout[-2000:]
    obs = json.loads(payload[-1])
    assert obs["action_dim"] == 14
    assert obs["action_token_begin_idx"] == 31743
    assert obs["stop_index"] == 2
    assert obs["ignore_index"] == -100
    assert obs["num_actions_chunk_unused"] == 25
    assert obs["proprio_dim_unused"] == 14


def test_legacy_entrypoint_holds_the_chunked_7d_defect():
    """Documents why the legacy eval loop must not be reused: it loops range(ACTION_DIM)."""
    src = open(os.path.join(REPO, "training", "train_pushT_pure2.py")).read()
    assert "range(ACTION_DIM)" in src
    assert "ProprioProjector(proprio_dim=14" in src


def test_r42_sources_avoid_prismatic_dimension_defaults():
    missing = [p for p in R42_SOURCES if not os.path.isfile(os.path.join(REPO, p))]
    assert not missing, "R42 sources not implemented yet: %s" % missing
    offenders = {}
    for rel in R42_SOURCES:
        text = open(os.path.join(REPO, rel)).read()
        hits = [pat for pat in DANGEROUS if pat in text]
        if hits:
            offenders[rel] = hits
    assert not offenders, "prismatic dimension defaults on the 14D path: %s" % offenders


def test_projector_width_guard_rejects_legacy_widths():
    for width in (7, 8, 14, 28, 80):
        with pytest.raises(R.R42Error):
            R.assert_projector_width(width, expected=42)
    assert R.assert_projector_width(42, expected=42) == 42


def test_episode_history_rejects_non_increasing_timestamps():
    meta = json.load(open(os.path.join(REPO, "evidence", "r9_gate_20261020",
                                      "frozen_contract_64train.json")))
    from r8_normalization import NormalizationContract
    h = R.EpisodeHistory(NormalizationContract.from_metadata(meta))
    s = np.zeros(14, np.float32)
    h.observe(0, s, timestamp=1.0)
    with pytest.raises(R.R42Error):
        h.observe(0, s, timestamp=1.0)      # equal timestamp = duplicate observation
    with pytest.raises(R.R42Error):
        h.observe(0, s, timestamp=0.5)      # backwards
