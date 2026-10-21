#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only tests for tools/m2_gate_check.py (gate P6_M2_TWO) and the trainer accounting that
feeds it (training/m2_train_shared.py).

Run:  /home/dgut/pusht_runtime/venv_oft/bin/python -m pytest tests/test_m2_gate_check.py -q

No GPU, no 7B model, no torch: the "model" is a fake logit generator over a synthetic HDF5 dataset,
so the PASS / FAIL / CANNOT_EVALUATE paths of the gate are all exercised for real.
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import h5py                                                        # noqa: E402

from training import m2_task_normalizer as TN                      # noqa: E402  (resolves M2)
from training import m2_train_shared as MT                         # noqa: E402
from tools import m2_gate_check as G                               # noqa: E402
import r42_adapter as R                                            # noqa: E402
import r42_train_contract as TC                                    # noqa: E402

M2 = TN.M2
TASKS = ("push_T", "stack_bowls")
SCALES = {"push_T": 1.0, "stack_bowls": 2.0}
VOCAB = R.TOKEN_BEGIN_IDX + 1 + 256 + 1        # just past the legal action window
PROMPT_IDS = {"push_T": [11, 12, 13], "stack_bowls": [11, 12, 13, 14, 15, 16]}


# -------------------------------------------------------------------------------- fixtures
def _jpeg(n=64):
    return b"\xff\xd8" + b"\x00" * max(0, n - 4) + b"\xff\xd9"


def write_episode(d, name, T=100, scale=1.0, seed=0, instruction=None):
    rng = np.random.RandomState(seed)
    S = np.cumsum(rng.randn(T, 14).astype(np.float64) * 0.05 * float(scale), axis=0)
    phase = np.arange(T, dtype=np.float64) * 0.1
    S[:, 6] = np.clip(0.5 + 0.2 * np.sin(phase) + 0.002 * scale * rng.randn(T), 0.0, 1.0)
    S[:, 13] = np.clip(0.5 + 0.2 * np.cos(phase) + 0.002 * scale * rng.randn(T), 0.0, 1.0)
    A = np.empty_like(S)
    A[:-1] = S[1:]
    A[-1] = S[-1]
    p = os.path.join(d, name)
    with h5py.File(p, "w") as h:
        for grp, arr in (("state", S), ("action", A)):
            i = 0
            for key, w in zip(("left_arm_joint_states", "left_ee_joint_states",
                               "right_arm_joint_states", "right_ee_joint_states"),
                              (6, 1, 6, 1)):
                h.create_dataset("%s/%s" % (grp, key), data=arr[:, i:i + w].astype(np.float64))
                i += w
        for grp in ("state", "action"):
            h.create_dataset("%s/left_ee_poses" % grp, data=rng.randn(T, 7))
            h.create_dataset("%s/right_ee_poses" % grp, data=rng.randn(T, 7))
        h.create_dataset("state/left_delta_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("state/right_delta_ee_poses", data=rng.randn(T, 7))
        h.create_dataset("additional_info/frequency", data=np.int64(25))
        h.create_dataset("data_format_version", data=np.bytes_("v1.0"))
        h.create_dataset("instruction", data=np.bytes_(instruction or M2.TASK_PROMPTS["push_T"]))
        for cam in M2.CAMERA_ORDER:
            g = h.create_group("vision/" + cam)
            g.create_dataset("colors", data=np.array([_jpeg()] * T, dtype="S64"))
            g.create_dataset("shape", data=np.array([480, 640, 3], dtype=np.int64))
            g.create_dataset("extrinsic_matrix", data=np.tile(np.eye(4), (T, 1, 1)))
            g.create_dataset("intrinsic_matrix", data=np.eye(3))
    return p


@pytest.fixture()
def ds(tmp_path):
    root = tmp_path / "hdf5"
    for task, scale in SCALES.items():
        d = root / task / "arx_x5" / "data"
        d.mkdir(parents=True)
        for i in range(5):
            write_episode(str(d), "episode_%07d.hdf5" % i, T=100, scale=scale,
                          seed=100 * M2.task_id(task) + i,
                          instruction=M2.TASK_PROMPTS[task])
    indexes = {t: M2.load_task_index(t, str(root)) for t in TASKS}
    splits = {t: M2.split_task_index(indexes[t], dev_frac=0.4, seed=0) for t in TASKS}
    payload, _provs = TN.fit_multi_normalizer(list(TASKS), splits, indexes=indexes)
    norm_path = str(tmp_path / "norm.json")
    TN.save_payload(norm_path, payload)
    splits_path = str(tmp_path / "data" / "m2_splits.json")
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as fh:
        json.dump({"schema": "m2_splits/v1", "splits": splits}, fh, indent=1)
    dev_frames = {t: MT.dev_frames_for_task(indexes[t], splits[t], t, 2, 0) for t in TASKS}
    return {"root": str(root), "indexes": indexes, "splits": splits, "payload": payload,
            "norm_path": norm_path, "splits_path": splits_path, "dev_frames": dev_frames}


# ------------------------------------------------------------------------------ fake model
def _logits(n_prompt, bins, n_supervised=15):
    L = n_prompt + n_supervised
    lg = np.full((L, VOCAB), -30.0, np.float32)
    for d in range(14):
        lg[n_prompt - 1 + d, R.TOKEN_BEGIN_IDX + 1 + int(bins[d])] = 30.0
    lg[n_prompt + 13, R.STOP_ID] = 30.0          # the supervised STOP position
    return lg


def fake_predict(ds, mode="good"):
    """mode: good (accurate + routed) | bad (inaccurate) | ignores_prompt (accurate, unrouted)."""
    payload = ds["payload"]

    def predict(prompt_task, frame_task, episode, t):
        d = ds["indexes"][frame_task]["hdf5_dir"]
        with h5py.File(os.path.join(d, episode), "r") as h:
            state = TC.read14(h, "state")
            action = TC.read14(h, "action")
        row = MT.row_for(payload, frame_task, state, action, t, episode=episode)
        lab = MT.labels_for(payload, frame_task, PROMPT_IDS[prompt_task], row["residual"])
        true_bins = MT.bins_from_residual(payload, frame_task, row["residual"])
        if mode == "bad":
            bins = (true_bins + 100) % 256
        elif mode == "ignores_prompt":
            bins = true_bins
        else:
            bins = true_bins if prompt_task == frame_task else (true_bins + 60) % 256
        return {"logits": _logits(lab["n_prompt"], bins),
                "labels": np.asarray(lab["labels"], np.int64),
                "n_prompt": int(lab["n_prompt"]),
                "residual": np.asarray(row["residual"], np.float32)}

    return predict


def build_archive(ds, tmp_path, mode="good", dev_frames=None):
    frames = dev_frames or ds["dev_frames"]
    blocks = MT.evaluate_dev_ce(fake_predict(ds, mode), ds["payload"], list(TASKS), frames)
    path = str(tmp_path / ("eval_%s.npz" % mode))
    MT.write_eval_archive(path, blocks)
    return path, blocks


def make_evidence(ds, ce_curves=None, max_tasks_stage=2, target=2000, reached=2000, start=0,
                  dev_frames=None, run_id="m2-test-run"):
    frames = dev_frames or ds["dev_frames"]
    curves = ce_curves or {t: [{"global_step": 0, "ce": 5.0}, {"global_step": 2000, "ce": 2.0}]
                           for t in TASKS}
    return {"schema": MT.SCHEMA_EVIDENCE, "run_id": run_id, "tasks": list(TASKS),
            "task_ids": {t: M2.TASK_IDS[t] for t in TASKS},
            "max_tasks_stage": int(max_tasks_stage),
            "target_global_step": int(target), "start_global_step": int(start),
            "global_step_reached": int(reached), "steps_executed": int(reached) - int(start),
            "dev_frames": {t: {"frames": [[ep, int(tt)] for ep, tt in frames[t]],
                               "frames_sha256": MT.dev_frames_sha256(frames[t])}
                           for t in TASKS},
            "ce_curves": curves, "ledger": {"bucket": MT.DEFAULT_BUCKET,
                                            "run_id": run_id, "steps_booked": None}}


def run_gate(ds, tmp_path, evidence, archive_path, normalizer=None):
    ev_path = str(tmp_path / "evidence.json")
    with open(ev_path, "w") as fh:
        json.dump(evidence, fh, indent=1)
    report, code = G.score(ev_path, archive_path, normalizer or ds["norm_path"],
                           splits_path=ds["splits_path"], hdf5_root=ds["root"])
    return report, code


def ids_failed(report):
    return {c["id"] for c in report["checks"] if not c["passed"]}


# ------------------------------------------------------------------------------- PASS path
def test_healthy_two_task_run_passes(ds, tmp_path):
    archive, blocks = build_archive(ds, tmp_path, "good")
    evidence = make_evidence(ds)
    report, code = run_gate(ds, tmp_path, evidence, archive)
    assert code == 0, (report["verdict"], ids_failed(report))
    assert report["gate"] == "P6_M2_TWO"
    assert report["verdict"] == "PASS" and report["exit"] == 0
    assert report["n_failed"] == 0
    for t in TASKS:
        pt = report["per_task"][t]
        assert pt["model_ce"] < pt["ce_noop"] < 5.6
        assert pt["margin_vs_noop"] > 0 and pt["margin_vs_cv"] > 0
        assert pt["n_frames"] == 2
    assert report["routing"]["min_own_prompt_win_rate"] == 1.0
    assert report["routing"]["min_wrong_prompt_ce_penalty"] > 0
    assert report["cap"] == 2000
    assert json.dumps(report), "the report must be JSON serialisable"


# ------------------------------------------------------------------------------- FAIL paths
def test_model_worse_than_both_baselines_fails(ds, tmp_path):
    archive, _ = build_archive(ds, tmp_path, "bad")
    report, code = run_gate(ds, tmp_path, make_evidence(ds), archive)
    assert code == 3 and report["verdict"] == "FAIL"
    failed = ids_failed(report)
    for t in TASKS:
        assert "P6.ce_vs_noop.%s" % t in failed
        assert "P6.ce_vs_constant_velocity.%s" % t in failed
        assert report["per_task"][t]["margin_vs_noop"] < 0


def test_prompt_ignoring_model_fails_routing_but_not_ce(ds, tmp_path):
    """The model predicts the truth for EVERY prompt: CE is perfect, routing is provably absent."""
    archive, _ = build_archive(ds, tmp_path, "ignores_prompt")
    report, code = run_gate(ds, tmp_path, make_evidence(ds), archive)
    assert code == 3 and report["verdict"] == "FAIL"
    failed = ids_failed(report)
    assert "P6.routing_not_degenerate.own_prompt_wins" in failed
    assert "P6.routing_not_degenerate.wrong_prompt_penalty" in failed
    for t in TASKS:
        assert "P6.ce_vs_noop.%s" % t not in failed, "the CE checks must still pass here"
    assert report["routing"]["min_wrong_prompt_ce_penalty"] == 0.0
    assert report["routing"]["min_own_prompt_win_rate"] == 0.0


def test_catastrophic_forgetting_fails(ds, tmp_path):
    archive, _ = build_archive(ds, tmp_path, "good")
    curves = {t: [{"global_step": 0, "ce": 5.0}, {"global_step": 1000, "ce": 2.0},
                  {"global_step": 2000, "ce": 5.6}] for t in TASKS}
    report, code = run_gate(ds, tmp_path, make_evidence(ds, ce_curves=curves), archive)
    assert code == 3
    failed = ids_failed(report)
    for t in TASKS:
        assert "P6.no_catastrophic_forgetting.%s" % t in failed
        assert report["forgetting"][t]["delta_worst_minus_step0"] > 0.10
        assert report["forgetting"][t]["best_ce"] == 2.0


def test_cap_violations_fail(ds, tmp_path):
    archive, _ = build_archive(ds, tmp_path, "good")
    report, code = run_gate(ds, tmp_path,
                            make_evidence(ds, target=2001, reached=2001), archive)
    assert code == 3 and "P6.cap.cumulative_steps" in ids_failed(report)
    report, code = run_gate(ds, tmp_path,
                            make_evidence(ds, max_tasks_stage=1, target=1000, reached=1000), archive)
    assert code == 3 and "P6.cap.task_count" in ids_failed(report)
    report, code = run_gate(ds, tmp_path,
                            make_evidence(ds, reached=1999, target=2000), archive)
    assert code == 3, "an unfinished stage is not a valid P6 verdict either"
    assert "P6.cap.stage_completed" in ids_failed(report)


def test_train_episode_in_the_dev_set_is_caught(ds, tmp_path):
    train_ep = ds["splits"]["push_T"]["train"][0].split("/", 1)[-1]
    leaky = {t: list(ds["dev_frames"][t]) for t in TASKS}
    leaky["push_T"] = [(train_ep, 10)]
    archive, _ = build_archive(ds, tmp_path, "good", dev_frames=leaky)
    report, code = run_gate(ds, tmp_path, make_evidence(ds, dev_frames=leaky), archive)
    assert code == 3 and "P6.dev_disjoint" in ids_failed(report)
    assert report["verdict"] == "FAIL"


# ---------------------------------------------------------------- CANNOT_EVALUATE paths
def test_missing_archive_cannot_evaluate(ds, tmp_path):
    report, code = run_gate(ds, tmp_path, make_evidence(ds), str(tmp_path / "nope.npz"))
    assert code == 4 and report["verdict"] == "CANNOT_EVALUATE"
    assert "inputs.consistent" in ids_failed(report)


def test_wrong_frames_cannot_evaluate(ds, tmp_path):
    archive, _ = build_archive(ds, tmp_path, "good")
    evidence = make_evidence(ds)
    evidence["dev_frames"]["push_T"]["frames_sha256"] = "0" * 64
    report, code = run_gate(ds, tmp_path, evidence, archive)
    assert code == 4 and "not the evidence DEV frames" in report["error"]


def test_archive_missing_a_task_cannot_evaluate(ds, tmp_path):
    blocks = MT.evaluate_dev_ce(fake_predict(ds, "good"), ds["payload"], list(TASKS),
                                ds["dev_frames"])
    one = {"push_T": blocks["push_T"]}
    path = str(tmp_path / "one_task.npz")
    MT.write_eval_archive(path, one)
    report, code = run_gate(ds, tmp_path, make_evidence(ds), path)
    assert code == 4 and "eval archive covers tasks" in report["error"]


def test_payload_missing_a_task_cannot_evaluate(ds, tmp_path):
    archive, _ = build_archive(ds, tmp_path, "good")
    p = copy.deepcopy(ds["payload"])
    for key in ("tasks", "task_id", "fitted_on"):
        del p[key]["stack_bowls"]
    bad_path = str(tmp_path / "norm_one.json")
    with open(bad_path, "w") as fh:
        json.dump(p, fh)
    report, code = run_gate(ds, tmp_path, make_evidence(ds), archive, normalizer=bad_path)
    assert code == 4
    assert "normalizer" in report["error"]


def test_tampered_true_bins_cannot_evaluate(ds, tmp_path):
    archive, blocks = build_archive(ds, tmp_path, "good")
    z = np.load(archive, allow_pickle=True)
    arrays = {k: z[k] for k in z.files}
    arrays["push_T/bins_true"] = (arrays["push_T/bins_true"] + 7) % 256
    tampered = str(tmp_path / "tampered.npz")
    np.savez_compressed(tampered, **arrays)
    report, code = run_gate(ds, tmp_path, make_evidence(ds), tampered)
    assert code == 4 and "true bins disagree" in report["error"]


def test_bad_schema_cannot_evaluate(ds, tmp_path):
    archive, _ = build_archive(ds, tmp_path, "good")
    evidence = make_evidence(ds)
    evidence["schema"] = "r15-evidence-v1"
    report, code = run_gate(ds, tmp_path, evidence, archive)
    assert code == 4 and "schema" in report["error"]


def test_cli_writes_the_report_and_returns_the_exit_code(ds, tmp_path, capsys):
    archive, _ = build_archive(ds, tmp_path, "good")
    ev_path = str(tmp_path / "evidence.json")
    with open(ev_path, "w") as fh:
        json.dump(make_evidence(ds), fh)
    out = str(tmp_path / "report.json")
    code = G.main(["--evidence", ev_path, "--eval", archive, "--normalizer", ds["norm_path"],
                   "--splits", ds["splits_path"], "--hdf5-root", ds["root"], "--out", out])
    assert code == 0
    report = json.load(open(out))
    assert report["gate"] == "P6_M2_TWO" and report["verdict"] == "PASS"
    assert report["exit"] == 0 and report["checks"]
    stdout = capsys.readouterr().out
    assert "VERDICT PASS" in stdout


# ------------------------------------------------------------------ trainer accounting
def test_task_mix_is_reproducible_balanced_and_floor_bounded():
    frames = {"push_T": 25313, "stack_bowls": 35858, "sweep_blocks": 47609}
    a = MT.plan_task_mix(list(frames), frames)
    b = MT.plan_task_mix(list(frames), frames)
    assert a["schedule_sha256"] == b["schedule_sha256"]
    assert a["policy"] == MT.MIX_POLICY
    assert sum(a["counts"].values()) == a["horizon"] == sum(frames.values())
    for t in frames:
        assert a["weights"][t] >= a["floor"] - 1e-9
        frac = a["counts"][t] / float(a["horizon"])
        assert abs(frac - a["weights"][t]) < 1e-3
    assert abs(sum(a["weights"].values()) - 1.0) < 1e-12
    assert MT.assert_mix_reproducible(a, frames) is True
    # the floor binds for the task with the fewest frames
    assert a["raw_weights"]["push_T"] < a["floor"]
    assert a["weights"]["push_T"] == pytest.approx(a["floor"])
    # the running mix never drifts: prefixes stay within one slot of the target
    prefix = {"push_T": 0, "stack_bowls": 0, "sweep_blocks": 0}
    for k, t in enumerate(a["schedule"][:5000], start=1):
        prefix[t] += 1
        for u in frames:
            assert abs(prefix[u] - k * a["weights"][u]) <= 1.0
    # a different seed does not change the schedule (order is fully deterministic)
    assert MT.plan_task_mix(list(frames), frames, seed=99)["schedule_sha256"] == \
        a["schedule_sha256"]
    # tampering with the recorded mix is refused
    bad = copy.deepcopy(a)
    bad["schedule_sha256"] = "0" * 64
    with pytest.raises(MT.M2TrainError):
        MT.assert_mix_reproducible(bad, frames)


def test_slot_frames_are_qualified_and_resumable(ds):
    frames_by_task = {}
    for t in TASKS:
        train = [k.split("/", 1)[-1] for k in ds["splits"][t]["train"]]
        frames_by_task[t] = [(ep, MM) for ep in train
                             for MM in range(M2.MIN_HISTORY_FRAMES, 6)]
    mix = MT.plan_task_mix(list(TASKS), {t: len(frames_by_task[t]) for t in TASKS}, horizon=64)
    slots = MT.build_slot_frames(mix, frames_by_task)
    assert len(slots) == 64
    for qual, _t in slots:
        task, _ep = M2.split_qualified(qual)
        assert task in TASKS
    sampler = MT.make_sampler(slots, 0)
    seen = [sampler.next() for _ in range(64)]
    assert sorted(seen) == sorted(slots), "one epoch must cover every slot exactly once"
    state = sampler.state()
    assert state["frames_sha256"] == state["frames_sha256"]
    again = MT.make_sampler(slots, 0)
    again.position = state["position"]
    assert again.order_hash() == sampler.order_hash()
    counts = MT.count_schedule_prefix(mix["schedule"], 10)
    assert sum(counts.values()) == 10


def test_cumulative_step_cap_is_enforced():
    assert MT.cumulative_step_cap(2) == 2000
    assert MT.cumulative_step_cap(1) == 1000
    assert MT.cumulative_step_cap(3) == 3000
    assert MT.assert_stage_caps(["push_T", "stack_bowls"], 2, 2000)["steps_requested"] == 2000
    assert MT.assert_stage_caps(["push_T", "stack_bowls"], 2, 1000, 500)["steps_requested"] == 500
    with pytest.raises(MT.M2TrainError):
        MT.assert_stage_caps(["push_T", "stack_bowls"], 2, 2001)
    with pytest.raises(MT.M2TrainError):
        MT.assert_stage_caps(["push_T", "stack_bowls", "sweep_blocks"], 2, 1000)
    with pytest.raises(MT.M2TrainError):
        MT.assert_stage_caps(["push_T", "stack_bowls"], 4, 1000)
    with pytest.raises(MT.M2TrainError):
        MT.assert_stage_caps(["push_T", "stack_bowls"], 2, 0)
    with pytest.raises(MT.M2TrainError):
        MT.assert_stage_caps(["push_T", "stack_blocks"], 2, 100)
    with pytest.raises(MT.M2TrainError):
        MT.assert_stage_caps(["push_T", "push_T"], 2, 100)
    assert MT.assert_stage_caps(["push_T"], 2, 500, 500)["already_at_target"] is True


def test_resume_guard_rejects_a_different_task_set_or_normalizer(ds):
    mix = MT.plan_task_mix(list(TASKS), {t: 1000 for t in TASKS}, horizon=100)
    sha = TN.normalizer_sha256(ds["payload"])
    ok_ts = {"tasks": list(TASKS), "normalizer_sha256": sha, "max_tasks_stage": 2,
             "task_mix": {"schedule_sha256": mix["schedule_sha256"]}}
    assert MT.assert_resume_compatible(ok_ts, list(TASKS), sha, mix, 2)["tasks"] == list(TASKS)
    with pytest.raises(MT.M2TrainError):
        MT.assert_resume_compatible(dict(ok_ts, tasks=["push_T"]), list(TASKS), sha, mix, 2)
    with pytest.raises(MT.M2TrainError):
        MT.assert_resume_compatible(dict(ok_ts, normalizer_sha256="0" * 64), list(TASKS), sha,
                                    mix, 2)
    other = MT.plan_task_mix(list(TASKS), {t: 1000 for t in TASKS}, horizon=101)
    with pytest.raises(MT.M2TrainError):
        MT.assert_resume_compatible(ok_ts, list(TASKS), sha, other, 2)
    with pytest.raises(MT.M2TrainError):
        MT.assert_resume_compatible({"tasks": list(TASKS)}, list(TASKS), sha, mix, 2)
    with pytest.raises(MT.M2TrainError):
        MT.assert_resume_compatible(dict(ok_ts, max_tasks_stage=3), list(TASKS), sha, mix, 2)


def test_prompts_are_task_distinct_and_guarded():
    prompts = MT.prompts_for(list(TASKS))
    assert len(set(prompts.values())) == 2
    for t in TASKS:
        assert M2.task_prompt(t) in prompts[t]
    assert MT.assert_prompts_distinct(list(TASKS), prompts)["prompt_sha256"]
    with pytest.raises(MT.M2TrainError):
        MT.assert_prompts_distinct(list(TASKS), {"push_T": prompts["push_T"],
                                                 "stack_bowls": prompts["push_T"]})
    with pytest.raises(MT.M2TrainError):
        MT.assert_prompts_distinct(list(TASKS), prompts,
                                   token_ids={"push_T": [1, 2], "stack_bowls": [1, 2]})
    with pytest.raises(MT.M2TrainError):
        MT.assert_prompts_distinct(list(TASKS), {"push_T": prompts["push_T"],
                                                 "stack_bowls": "Push it."})


def test_teacher_forced_ce_math_is_exact():
    V = VOCAB
    labels = [5, 6, 7] + [R.TOKEN_BEGIN_IDX + 1 + d for d in range(14)] + [R.STOP_ID]
    n_prompt = 3
    L = len(labels)
    assert L == n_prompt + 15
    flat = np.zeros((L, V), np.float32)                 # uniform -> CE = ln(V) at every position
    out = MT.token_ce_from_logits(flat, labels, n_prompt)
    assert out["ce"] == pytest.approx(float(np.log(V)), rel=1e-5)
    assert out["n_supervised"] == 15
    hot = np.full((L, V), -40.0, np.float32)
    for i, tok in enumerate(labels[n_prompt:]):
        hot[n_prompt - 1 + i, tok] = 40.0
    assert MT.token_ce_from_logits(hot, labels, n_prompt)["ce"] < 1e-3
    win = MT.action_window_probs(flat, n_prompt)
    assert win["pred_probs"].shape == (14, 256)
    assert np.allclose(win["pred_probs"].sum(axis=1), 1.0)
    assert MT.ce_window_from_probs(win["pred_probs"], np.zeros(14, int)) == pytest.approx(
        float(np.log(256)), rel=1e-5)
    good = np.zeros((14, 256), np.float32)
    good[:, 7] = 1.0
    assert MT.ce_window_from_probs(good, np.full(14, 7)) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(MT.M2TrainError):
        MT.token_ce_from_logits(flat, labels, n_prompt, n_supervised=99)
    with pytest.raises(MT.M2TrainError):
        MT.ce_window_from_probs(win["pred_probs"], np.array([999] + [0] * 13))
