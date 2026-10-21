#!/usr/bin/env python3
"""R16 G4: the r42_metadata.json bridge for R15 stage checkpoints.

The load-bearing test is test_byte_parity_against_existing_metadata: it rebuilds the metadata of
every checkpoint in runs/ that already has an r42_metadata.json (written by run_smoke/run_overfit)
and requires the produced FILE to be byte-identical.  If that passes, the bridge is not an
approximation of the training writer - it is the same writer fed the same evidence.

The remaining tests use tiny synthetic checkpoints so the failure modes (a tampered file, a
contract that is not 42D signed_motion, disagreeing step counters) are cheap to exercise.
"""
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402
import r42_metadata_bridge as B  # noqa: E402

CONTRACT = os.path.join(REPO, "evidence", "r9_gate_20261020", "frozen_contract_64train.json")
BOUNDS = os.path.join(REPO, "data", "push_T_bounds.npz")
R15_STEP300 = os.path.join(REPO, "runs", "r15_stage_300_20261021T030423Z", "checkpoints",
                           "step300")
EXISTING = sorted(glob.glob(os.path.join(REPO, "runs", "*", "checkpoints", "step*",
                                         "r42_metadata.json")))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------ the load-bearing test
@pytest.mark.skipif(not EXISTING, reason="no pre-existing r42_metadata.json in runs/ to compare")
def test_byte_parity_against_existing_metadata(tmp_path):
    """Every metadata file the training writers produced must be reproduced byte for byte."""
    assert len(EXISTING) >= 3, "expected the r13/r14 references, found %d" % len(EXISTING)
    for reference in EXISTING:
        checkpoint = os.path.dirname(reference)
        step = int(os.path.basename(checkpoint).replace("step", ""))
        out = os.path.join(str(tmp_path), os.path.basename(checkpoint),
                           "r42_metadata.json")
        meta, report = B.build_stage_metadata(checkpoint, train_steps=step)
        assert report["train_steps"] == step
        R.validate_metadata(meta, expect_task="push_T")
        assert report["lora_sha256"] == _sha(os.path.join(
            checkpoint, "lora_adapter", "adapter_model.safetensors"))
        B.write_metadata(meta, out)
        result = B.assert_byte_parity(out, reference)
        assert result["bytes"] == os.path.getsize(reference)
        assert result["sha256"] == _sha(reference)


@pytest.mark.skipif(not os.path.isdir(R15_STEP300),
                    reason="R15 stage checkpoint not present on this box")
def test_bridge_serves_an_r15_stage_checkpoint_that_has_no_metadata(tmp_path):
    """The actual G4 case: MANIFEST.json + trainer_state.json + contract + bounds only."""
    assert not os.path.isfile(os.path.join(R15_STEP300, "r42_metadata.json"))
    meta, report = B.build_stage_metadata(R15_STEP300)
    assert report["manifest_source"] == "manifest"
    assert report["manifest_schema"] == "r15-full-state-v1"
    assert report["train_steps"] == 300
    assert report["base_model_source"] == "trainer_state.base_model.config.json"
    assert meta["train_steps"] == 300 and meta["task_id"] == "push_T"
    assert meta["codec_mode"] == "fixed" and meta["proprio_dim"] == 42
    manifest = json.load(open(os.path.join(R15_STEP300, "MANIFEST.json")))
    assert meta["lora_sha256"] == manifest["files"]["lora_adapter/adapter_model.safetensors"]
    assert meta["projector_sha256"] == manifest["files"]["proprio_projector_42.pt"]
    trainer = json.load(open(os.path.join(R15_STEP300, "trainer_state.json")))
    assert meta["base_model_sha256"] == trainer["base_model"]["config.json"]
    assert meta["normalizer_sha256"] == \
        trainer["data_contract"]["normalizer_sha256"]
    assert meta["frozen_contract_sha256"] == _sha(CONTRACT)
    out = os.path.join(str(tmp_path), "r42_metadata.json")
    info = B.write_metadata(meta, out)
    assert info["bytes"] > 4000
    assert json.load(open(out)) == json.loads(R.canonical_json(meta))


# --------------------------------------------------------------------- synthetic checkpoints
def _tiny_checkpoint(root, lora=b"lora-bytes", projector=b"proj-bytes", global_step=5,
                     manifest=True, trainer=True, base_sha=None):
    ck = os.path.join(str(root), "step5")
    os.makedirs(os.path.join(ck, "lora_adapter"), exist_ok=True)
    lora_path = os.path.join(ck, "lora_adapter", "adapter_model.safetensors")
    proj_path = os.path.join(ck, "proprio_projector_42.pt")
    with open(lora_path, "wb") as fh:
        fh.write(lora)
    with open(proj_path, "wb") as fh:
        fh.write(projector)
    if manifest:
        with open(os.path.join(ck, "MANIFEST.json"), "w") as fh:
            json.dump({"schema": "r15-full-state-v1", "global_step": global_step,
                       "files": {"lora_adapter/adapter_model.safetensors": _sha(lora_path),
                                 "proprio_projector_42.pt": _sha(proj_path)}}, fh)
    if trainer:
        payload = {"schema": "r15-full-state-v1", "global_step": global_step}
        if base_sha:
            payload["base_model"] = {"config.json": base_sha}
        with open(os.path.join(ck, "trainer_state.json"), "w") as fh:
            json.dump(payload, fh)
        if manifest:
            raw = json.load(open(os.path.join(ck, "MANIFEST.json")))
            raw["files"]["trainer_state.json"] = _sha(os.path.join(ck, "trainer_state.json"))
            with open(os.path.join(ck, "MANIFEST.json"), "w") as fh:
                json.dump(raw, fh)
    return ck


BASE_SHA = "a" * 64


def test_minimal_checkpoint_builds_and_validates(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    meta, report = B.build_stage_metadata(ck)
    assert report["manifest_source"] == "manifest"
    assert report["train_steps"] == 5
    assert report["base_model_source"] == "trainer_state.base_model.config.json"
    assert meta["lora_sha256"] == hashlib.sha256(b"lora-bytes").hexdigest()
    assert meta["projector_sha256"] == hashlib.sha256(b"proj-bytes").hexdigest()
    assert meta["train_steps"] == 5
    R.validate_metadata(meta, expect_task="push_T")


def test_checkpoint_without_manifest_or_trainer_state_uses_the_files(tmp_path):
    ck = _tiny_checkpoint(tmp_path, manifest=False, trainer=False)
    base_cfg = os.path.join(str(tmp_path), "config.json")
    with open(base_cfg, "w") as fh:
        fh.write("{}")
    meta, report = B.build_stage_metadata(ck, train_steps=5, base_model_config=base_cfg)
    assert report["manifest_source"] == "derived"
    assert report["base_model_source"] == "sha256(config.json)"
    assert meta["base_model_sha256"] == _sha(base_cfg)


def test_tampered_checkpoint_file_is_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    with open(os.path.join(ck, "proprio_projector_42.pt"), "wb") as fh:
        fh.write(b"tampered")
    with pytest.raises(B.R42BridgeError, match="MANIFEST/file disagreement"):
        B.build_stage_metadata(ck)


def test_missing_projector_is_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    os.remove(os.path.join(ck, "proprio_projector_42.pt"))
    with pytest.raises(B.R42BridgeError, match="missing"):
        B.build_stage_metadata(ck)


def test_disagreeing_step_counters_are_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA, global_step=5)
    with pytest.raises(B.R42BridgeError, match="train_steps disagreement"):
        B.build_stage_metadata(ck, train_steps=7)


def test_no_base_model_attestation_is_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, manifest=False, trainer=False)
    with pytest.raises(B.R42BridgeError, match="no base-model attestation"):
        B.build_stage_metadata(ck, train_steps=5, base_model_config=None)


def test_modified_trainer_state_is_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    path = os.path.join(ck, "trainer_state.json")
    raw = json.load(open(path))
    raw["base_model"] = {"config.json": "b" * 64}
    with open(path, "w") as fh:
        json.dump(raw, fh)
    with pytest.raises(B.R42BridgeError, match="trainer_state.json does not match"):
        B.build_stage_metadata(ck)


def test_contract_that_is_not_the_42d_contract_is_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    bad = json.load(open(CONTRACT))
    bad["normalization_mode"] = "per_dim_minmax"
    path = os.path.join(str(tmp_path), "contract.json")
    with open(path, "w") as fh:
        json.dump(bad, fh)
    with pytest.raises(B.R42BridgeError, match="normalization_mode"):
        B.build_stage_metadata(ck, contract_path=path)


def test_contract_with_legacy_gripper_dims_is_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    bad = json.load(open(CONTRACT))
    bad["gripper_policy"] = {"gripper_dims": [6, 7], "model_dims": [], "rule": "x"}
    path = os.path.join(str(tmp_path), "contract.json")
    with open(path, "w") as fh:
        json.dump(bad, fh)
    with pytest.raises(B.R42BridgeError, match="gripper_dims"):
        B.build_stage_metadata(ck, contract_path=path)


def test_contract_sha_disagreeing_with_the_trainer_state_is_refused(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    path = os.path.join(ck, "trainer_state.json")
    raw = json.load(open(path))
    raw["data_contract"] = {"frozen_contract_sha256": "c" * 64}
    with open(path, "w") as fh:
        json.dump(raw, fh)
    manifest = json.load(open(os.path.join(ck, "MANIFEST.json")))
    manifest["files"]["trainer_state.json"] = _sha(path)
    with open(os.path.join(ck, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh)
    with pytest.raises(B.R42BridgeError, match="frozen contract sha256 mismatch"):
        B.build_stage_metadata(ck)


def test_existing_metadata_is_never_overwritten(tmp_path):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    B.build_and_write(ck, base_model_config=None)
    with pytest.raises(B.R42BridgeError, match="refusing to overwrite"):
        B.build_and_write(ck, base_model_config=None)


def test_boundary_writer_and_bridge_write_the_same_bytes(tmp_path):
    """The bridge must use r42_adapter.save_metadata, not its own json.dump."""
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    meta, _report = B.build_stage_metadata(ck)
    a = os.path.join(str(tmp_path), "a.json")
    b = os.path.join(str(tmp_path), "b.json")
    B.write_metadata(meta, a)
    R.save_metadata(b, meta)
    assert open(a, "rb").read() == open(b, "rb").read()


def test_cli_refuses_and_reports_a_bad_checkpoint(tmp_path, capsys):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    with open(os.path.join(ck, "lora_adapter", "adapter_model.safetensors"), "wb") as fh:
        fh.write(b"changed")
    code = B.main(["--checkpoint", ck, "--out", os.path.join(str(tmp_path), "x.json")])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().out


def test_cli_writes_and_prints_the_provenance(tmp_path, capsys):
    ck = _tiny_checkpoint(tmp_path, base_sha=BASE_SHA)
    out = os.path.join(str(tmp_path), "meta", "r42_metadata.json")
    code = B.main(["--checkpoint", ck, "--out", out])
    assert code == 0
    text = capsys.readouterr().out
    assert "wrote" in text and "codec_mode=fixed" in text
    assert os.path.isfile(out)
