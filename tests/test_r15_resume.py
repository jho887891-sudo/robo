#!/usr/bin/env python3
"""R15 Lane T / P0-RESUME-1 regression: N continuous steps must equal K + save + load + (N-K).

Runs entirely on CPU with a tiny torch model, so it is a real equivalence test of the resume
machinery (weights, AdamW moments, scheduler, all three RNGs, sampler position, global_step)
without loading the 7B.
"""
import json
import os
import random
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))

import r15_resume as RS  # noqa: E402

FRAMES = [("episode_%04d.hdf5" % i, 10 + i) for i in range(12)]
CONTRACT = {"split_sha256": "s" * 64, "frozen_contract_sha256": "c" * 64,
            "hdf5_dir": "/data/push_T", "codec": {"mode": "fixed", "window": [31744, 31999],
                                                  "stop": 2},
            "normalizer_sha256": "n" * 64, "cams": 3, "lr": 5e-4, "seed": 7}


def build():
    RS.seed_everything(CONTRACT["seed"])
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    proj = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(list(model.parameters()) + list(proj.parameters()), lr=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6], gamma=0.1)
    sampler = RS.FrameSampler(FRAMES, CONTRACT["seed"])
    return model, proj, opt, sched, sampler


def one_step(model, proj, opt, sched, sampler):
    """A step that deliberately consumes python/numpy/torch RNG and the sampler."""
    ep, t = sampler.next()
    x = torch.randn(3, 8) * (1.0 + 0.01 * random.random()) + 0.001 * float(np.random.rand())
    y = torch.full((3, 4), float(t % 7) / 7.0)
    opt.zero_grad(set_to_none=True)
    loss = ((model(x) + proj(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    sched.step()
    return float(loss)


def run_steps(n, model, proj, opt, sched, sampler):
    for _ in range(n):
        one_step(model, proj, opt, sched, sampler)


def snapshot(model, proj, opt, sched, sampler):
    return {"model": [p.detach().clone() for p in model.parameters()],
            "proj": [p.detach().clone() for p in proj.parameters()],
            "exp_avg": [s["exp_avg"].clone() for s in opt.state_dict()["state"].values()],
            "lr": sched.get_last_lr()[0],
            "step": sched.state_dict()["last_epoch"],
            "sampler_pos": sampler.position,
            "next_sample": sampler.perm[sampler.position % len(sampler.perm)]}


def test_continuous_equals_resume(tmp_path):
    # --- continuous 12 steps
    m1, p1, o1, s1, sm1 = build()
    run_steps(12, m1, p1, o1, s1, sm1)
    want = snapshot(m1, p1, o1, s1, sm1)

    # --- 5 steps, checkpoint, fresh objects, resume, 7 steps
    m2, p2, o2, s2, sm2 = build()
    run_steps(5, m2, p2, o2, s2, sm2)
    ck = RS.save_full_state(str(tmp_path / "ckpt_step5"), model=m2, proj=p2, opt=o2, sched=s2,
                            sampler=sm2, global_step=5, run_id="run-a",
                            data_contract=CONTRACT, model_kind="torch")

    m3, p3, o3, s3, sm3 = build()          # deliberately different random init
    out = RS.load_full_state(ck["dir"], model=m3, proj=p3, opt=o3, sched=s3, frames=FRAMES,
                             model_kind="torch", expected_contract=CONTRACT,
                             expected_global_step=5)
    assert out["global_step"] == 5
    sm3 = out["sampler"]                      # the RESTORED sampler, not the fresh one
    assert sm3.position == 5
    run_steps(7, m3, p3, o3, s3, sm3)
    got = snapshot(m3, p3, o3, s3, sm3)

    for a, b in zip(want["model"], got["model"]):
        assert torch.equal(a, b), "model weights diverged after resume"
    for a, b in zip(want["proj"], got["proj"]):
        assert torch.equal(a, b), "projector diverged after resume"
    for a, b in zip(want["exp_avg"], got["exp_avg"]):
        assert torch.equal(a, b), "AdamW moments diverged after resume"
    assert want["lr"] == got["lr"], "LR schedule diverged"
    assert want["step"] == got["step"]
    assert want["sampler_pos"] == got["sampler_pos"] == 12
    assert want["next_sample"] == got["next_sample"], "next sample differs after resume"


def test_resume_without_state_does_not_match(tmp_path):
    """Control: re-initialising AdamW (the old behaviour) must NOT be equivalent."""
    m1, p1, o1, s1, sm1 = build()
    run_steps(12, m1, p1, o1, s1, sm1)
    want = snapshot(m1, p1, o1, s1, sm1)
    m2, p2, o2, s2, sm2 = build()
    run_steps(5, m2, p2, o2, s2, sm2)
    RS.save_full_state(str(tmp_path / "ck"), model=m2, proj=p2, opt=o2, sched=s2, sampler=sm2,
                       global_step=5, run_id="run-b", data_contract=CONTRACT, model_kind="torch")
    m3, p3, o3, s3, sm3 = build()
    # old behaviour: load only weights, then create a fresh optimizer/scheduler
    ckdir = str(tmp_path / "ck")
    m3.load_state_dict(torch.load(os.path.join(ckdir, "model_state.pt"), map_location="cpu"))
    p3.load_state_dict(torch.load(os.path.join(ckdir, "proprio_projector_42.pt"),
                                  map_location="cpu"))
    for _ in range(7):
        one_step(m3, p3, o3, s3, sm3)
    got = snapshot(m3, p3, o3, s3, sm3)
    assert any(not torch.allclose(a, b, atol=1e-8)
               for a, b in zip(want["model"], got["model"])), \
        "weights-only reload must not reproduce the continuous path"


def test_save_never_overwrites(tmp_path):
    m, p, o, s, sm = build()
    d = str(tmp_path / "ck")
    RS.save_full_state(d, model=m, proj=p, opt=o, sched=s, sampler=sm, global_step=0,
                       run_id="r", data_contract=CONTRACT, model_kind="torch")
    with pytest.raises(RS.ResumeError):
        RS.save_full_state(d, model=m, proj=p, opt=o, sched=s, sampler=sm, global_step=1,
                           run_id="r", data_contract=CONTRACT, model_kind="torch")


def test_failed_save_leaves_no_directory(tmp_path):
    m, p, o, s, sm = build()
    with pytest.raises(RS.ResumeError):
        RS.save_full_state(str(tmp_path / "ck"), model=m, proj=p, opt=o, sched=s, sampler=sm,
                           global_step=0, run_id="r", data_contract=CONTRACT,
                           model_kind="bogus")
    assert not os.path.exists(tmp_path / "ck")
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]


def _save(tmp_path, **over):
    m, p, o, s, sm = build()
    RS.seed_everything(7)
    run_steps(1, m, p, o, s, sm)
    c = dict(CONTRACT)
    c.update(over.pop("contract", {}))
    d = str(tmp_path / ("ck_" + over.pop("name", "0")))
    RS.save_full_state(d, model=m, proj=p, opt=o, sched=s, sampler=sm,
                       global_step=over.pop("global_step", 1), run_id="r", data_contract=c,
                       model_kind="torch")
    return d


def _load(d, frames=None):
    m, p, o, s, sm = build()
    return RS.load_full_state(d, model=m, proj=p, opt=o, sched=s, frames=frames or FRAMES,
                              model_kind="torch")


def test_corrupt_file_is_refused(tmp_path):
    d = _save(tmp_path)
    with open(os.path.join(d, "scheduler_state.pt"), "wb") as fh:
        fh.write(b"not a checkpoint")
    with pytest.raises(RS.ResumeError):
        _load(d)


def test_missing_file_is_refused(tmp_path):
    d = _save(tmp_path)
    os.unlink(os.path.join(d, "rng.pt"))
    with pytest.raises(RS.ResumeError):
        _load(d)


def test_wrong_parent_global_step_is_refused(tmp_path):
    d = _save(tmp_path, global_step=300)
    m, p, o, s, sm = build()
    with pytest.raises(RS.ResumeError):
        RS.load_full_state(d, model=m, proj=p, opt=o, sched=s, frames=FRAMES,
                           model_kind="torch", expected_global_step=1000)


def test_changed_data_contract_is_refused(tmp_path):
    d = _save(tmp_path)
    m, p, o, s, sm = build()
    changed = json.loads(json.dumps(CONTRACT))
    changed["normalizer_sha256"] = "x" * 64
    with pytest.raises(RS.ResumeError):
        RS.load_full_state(d, model=m, proj=p, opt=o, sched=s, frames=FRAMES,
                           model_kind="torch", expected_contract=changed)


def test_changed_sampler_frames_is_refused(tmp_path):
    d = _save(tmp_path)
    with pytest.raises(RS.ResumeError):
        _load(d, frames=FRAMES[:-1] + [("episode_9999.hdf5", 3)])


def test_legacy_schema_is_refused(tmp_path):
    d = _save(tmp_path)
    man = json.load(open(os.path.join(d, "MANIFEST.json")))
    man["schema"] = "r14-lora-only-v0"
    json.dump(man, open(os.path.join(d, "MANIFEST.json"), "w"))
    with pytest.raises(RS.ResumeError):
        _load(d)


def test_wrong_optimizer_format_version_is_refused(tmp_path):
    d = _save(tmp_path)
    ts = json.load(open(os.path.join(d, "trainer_state.json")))
    ts["optimizer_format_version"] = 99
    json.dump(ts, open(os.path.join(d, "trainer_state.json"), "w"))
    man = json.load(open(os.path.join(d, "MANIFEST.json")))
    man["files"]["trainer_state.json"] = RS.sha256_file(os.path.join(d, "trainer_state.json"))
    json.dump(man, open(os.path.join(d, "MANIFEST.json"), "w"))
    with pytest.raises(RS.ResumeError):
        _load(d)


def test_seed_before_init_is_reproducible():
    """The audit's ordering fix: seeding BEFORE construction makes the fresh start reproducible."""
    def make(seed):
        RS.seed_everything(seed)
        m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))
        return [p.detach().clone() for p in m.parameters()]

    a, b, c = make(7), make(7), make(8)
    for x, y in zip(a, b):
        assert torch.equal(x, y), "same seed must give the same fresh initialisation"
    assert any(not torch.equal(x, z) for x, z in zip(a, c)), \
        "a different seed must change the initialisation (seed is really applied pre-init)"


def test_peeking_the_next_frame_does_not_consume_it():
    """PR#21 finding 1: logging the next frame must not advance the sampler."""
    sm = RS.FrameSampler(FRAMES, 3)
    peek = sm.perm[sm.position % len(sm.perm)]
    assert sm.frames[peek] == sm.next()
    sm2 = RS.FrameSampler(FRAMES, 3)
    peek2 = sm2.perm[sm2.position % len(sm2.perm)]
    assert sm2.position == 0
    assert sm2.frames[peek2] == sm2.next()

def test_sampler_is_permutation_and_resumable():
    sm = RS.FrameSampler(FRAMES, 3)
    first = [sm.next() for _ in range(len(FRAMES))]
    assert sorted(first) == sorted(FRAMES)
    st = sm.state()
    sm2 = RS.FrameSampler.from_state(FRAMES, st)
    assert sm2.state() == st
    assert sm2.next() == sm.next(), "resumed sampler must yield the same next frame"
    with pytest.raises(RS.ResumeError):
        RS.FrameSampler.from_state(FRAMES[:-1], st)
