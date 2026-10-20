"""CPU contract tests for the R8 proposal: 42-dim history proprio + velocity residual.

These tests do not train anything.  They pin the interface that a 42-dim
ProprioProjector, the 256-bin tokenizer and the KV-cache autoregressive decode
would all have to honour, and they run against the real push-T HDF5 episodes so
that the contract is checked on the actual data shape and value range.

Contract under test
    X_t   = [S_t, V_t, Acc_t]                      42 dims, past-only
    R_t   = (A_t - S_t) - V_t
    A_hat = S_t + V_t + R_hat
    boundary: states before the episode start are clamped to S_0 (rest at start)
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "diagnostics"))
sys.path.insert(0, HERE)

import r6_data as D                                    # noqa: E402
import r8_velocity_residual_contract as C              # noqa: E402

SPLIT = os.environ.get("PUSHT_SPLIT", "/home/dgut/pusht_runtime/split80.json")
HDF5 = os.environ.get("PUSHT_HDF5", "")


def _split():
    return json.load(open(SPLIT))


def _episode(name):
    return D.load_episode(name)


def _legal_states(ep, limit=None):
    t = D.valid_frames(len(ep["state"]))
    if limit:
        t = t[:limit]
    return ep["state"][t], ep["action"][t]


def _synthetic_states(n=7, dim=14, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=0.05, size=(n, dim)), axis=0)


# ------------------------------------------------------------------ shape/layout
def test_proprio42_shape_and_slot_layout():
    states = _synthetic_states(9)
    X = C.build_proprio42(states)
    assert X.shape == (9, 42)
    assert C.PROPRIO42_DIM == 42
    assert np.allclose(X[:, 0:14], states)
    assert np.allclose(X[:, 14:28], C.velocity(states))
    assert np.allclose(X[:, 28:42], C.acceleration(states))


def test_history_time_alignment_is_exact_past_finite_difference():
    states = _synthetic_states(11, seed=3)
    V = C.velocity(states)
    Acc = C.acceleration(states)
    for t in range(2, len(states)):
        assert np.allclose(V[t], states[t] - states[t - 1])
        assert np.allclose(Acc[t], states[t] - 2 * states[t - 1] + states[t - 2])
    # and the two quantities are consistent: Acc_t = V_t - V_(t-1)
    for t in range(2, len(states)):
        assert np.allclose(Acc[t], V[t] - V[t - 1])


def test_boundary_first_and_second_frame_follow_the_rest_at_start_rule():
    states = _synthetic_states(5, seed=7)
    V = C.velocity(states)
    Acc = C.acceleration(states)
    assert np.allclose(V[0], 0.0), "first frame must imply zero velocity"
    assert np.allclose(Acc[0], 0.0), "first frame must imply zero acceleration"
    assert np.allclose(V[1], states[1] - states[0])
    assert np.allclose(Acc[1], states[1] - states[0]), "second frame clamps S_{-1} to S_0"


def test_no_future_information_leaks_into_the_current_input():
    states = _synthetic_states(8, seed=11)
    X = C.build_proprio42(states)
    cut = 4
    mutated = states.copy()
    mutated[cut + 1:] += 1000.0
    X2 = C.build_proprio42(mutated)
    assert np.array_equal(X[: cut + 1], X2[: cut + 1]), "X_t must not depend on S_{>t}"
    assert not np.allclose(X[cut + 2:], X2[cut + 2:])


# --------------------------------------------------------------- target/recovery
def test_velocity_residual_target_definition():
    states = _synthetic_states(6, seed=13)
    actions = states + 0.01
    R = C.velocity_residual(states, actions)
    for t in range(len(states)):
        V = np.zeros(14) if t == 0 else states[t] - states[t - 1]
        assert np.allclose(R[t], (actions[t] - states[t]) - V)


def test_recovery_with_true_residual_reproduces_the_action_exactly():
    states = _synthetic_states(6, seed=17)
    actions = states + 0.02
    R = C.velocity_residual(states, actions)
    A_hat = C.recover_action(states, R)
    assert np.allclose(A_hat, actions, atol=1e-12), "A = S + V + R must be exact"


def test_zero_residual_recovers_constant_velocity_and_never_double_subtracts():
    states = _synthetic_states(6, seed=19)
    zeros = np.zeros_like(states)
    A_hat = C.recover_action(states, zeros)
    V = C.velocity(states)
    assert np.allclose(A_hat, states + V)
    # the historical bug subtracted V a second time; that would equal S here
    assert not np.allclose(A_hat, states) or np.allclose(V, 0.0)


# ------------------------------------------------------------- tokenizer contract
def test_token_roundtrip_stays_within_the_documented_quantisation_bound():
    codec = C.load_codec()
    split = _split()
    train_R = []
    for name in split["train"][:8]:
        s, a = _legal_states(_episode(name))
        train_R.append(C.velocity_residual(s, a))
    train_R = np.concatenate(train_R)
    q01, q99 = C.fit_bounds_on_train(train_R)
    probe_R = train_R[:500]
    back, tokens = C.token_roundtrip(probe_R, q01, q99, codec)
    bound = C.quantization_error_bound(q01, q99)
    # values outside the training window are intentionally clipped by the codec,
    # so the bound is stated against the clipped target, not the raw residual
    target = C.clip_to_window(probe_R, q01, q99)
    err = np.abs(back - target)
    assert np.all(err <= bound[None, :] + 1e-12), "decode error exceeds 2/255 of the window"
    assert codec.validate_token_ids(np.asarray(tokens).reshape(-1)), "illegal action token produced"


def test_channel_window_coverage_is_measured_not_assumed():
    split = _split()
    train_R = []
    for name in split["train"][:8]:
        s, a = _legal_states(_episode(name))
        train_R.append(C.velocity_residual(s, a))
    train_R = np.concatenate(train_R)
    q01, q99 = C.fit_bounds_on_train(train_R)
    cov = C.window_coverage(train_R, q01, q99)
    assert len(cov["coverage_fraction"]) == 14
    assert all(0.0 <= c <= 1.0 for c in cov["coverage_fraction"])
    # the q01..q99 window must cover at least the bulk of the training residual
    assert min(cov["coverage_fraction"]) >= 0.90, cov
    # and the report must expose the channels where the residual exceeds the window
    worst = int(np.argmax(cov["truncation_ratio"]))
    assert cov["truncation_ratio"][worst] > 1.0


def test_degenerate_channel_windows_are_widened_explicitly():
    flat = np.zeros((50, 14))
    flat[:, 0] = np.linspace(-1e-3, 1e-3, 50)
    lo, hi = C.fit_bounds_on_train(flat)
    assert (hi - lo > 0).all(), "a collapsed channel must be widened, not left degenerate"
    assert hi[3] - lo[3] == pytest.approx(2.0e-6, rel=0.2)


def test_quantised_residual_still_recovers_the_action_within_the_bound():
    codec = C.load_codec()
    split = _split()
    train_R = []
    for name in split["train"][:8]:
        s, a = _legal_states(_episode(name))
        train_R.append(C.velocity_residual(s, a))
    q01, q99 = C.fit_bounds_on_train(np.concatenate(train_R))
    states, actions = _legal_states(_episode(split["train"][0]), limit=400)
    R = C.velocity_residual(states, actions)
    back, _ = C.token_roundtrip(R, q01, q99, codec)
    A_hat = C.recover_action(states, back)
    target = C.recover_action(states, C.clip_to_window(R, q01, q99))
    err = np.abs(A_hat - target)
    assert np.all(err <= C.quantization_error_bound(q01, q99)[None, :] + 1e-9)


# --------------------------------------------------------- legacy / deploy parity
def test_legacy_14d_path_still_exists_and_42d_rejects_wrong_width():
    sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))
    import proprio_history as PH

    assert PH.proprio_dim(PH.NONE) == 14
    assert PH.proprio_dim(PH.TRUE) == 28
    with pytest.raises(ValueError):
        C.build_proprio42(np.zeros((4, 28)))


def test_streaming_deployment_matches_batched_training_construction():
    states = _synthetic_states(9, seed=23)
    batched = C.build_proprio42(states)
    history = C.ObservedHistory()
    streamed = []
    for t in range(len(states)):
        history.push(states[t])
        window = history.window()
        X = C.build_proprio42(window[-3:])[-1]
        streamed.append(X)
    streamed = np.vstack(streamed)
    assert np.allclose(streamed, batched), "deployment and training must build identical X_t"


def test_history_reset_drops_the_previous_episode():
    history = C.ObservedHistory()
    history.push(np.zeros(14))
    history.push(np.ones(14))
    history.reset()
    history.push(np.full(14, 5.0))
    X = C.build_proprio42(history.window()[-3:])[-1]
    assert np.allclose(X[0:14], 5.0)
    assert np.allclose(X[14:28], 0.0), "after reset the implied velocity must be zero"


# ------------------------------------------------------------------- real dataset
@pytest.mark.skipif(not os.path.isfile(HDF5 + "/" + _split()["train"][0]) if HDF5 else True,
                    reason="PUSHT_HDF5 not configured")
def test_real_hdf5_minibatch_end_to_end():
    codec = C.load_codec()
    split = _split()
    ep = _episode(split["train"][0])
    states, actions = _legal_states(ep, limit=64)
    X = C.build_proprio42(states)
    R = C.velocity_residual(states, actions)
    assert X.shape == (64, 42) and R.shape == (64, 14)
    assert np.isfinite(X).all() and np.isfinite(R).all()

    train_R = []
    for name in _split()["train"][:8]:
        s, a = _legal_states(_episode(name))
        train_R.append(C.velocity_residual(s, a))
    q01, q99 = C.fit_bounds_on_train(np.concatenate(train_R))
    back, tokens = C.token_roundtrip(R, q01, q99, codec)
    A_hat = C.recover_action(states, back)
    target = C.recover_action(states, C.clip_to_window(R, q01, q99))
    assert np.all(np.abs(A_hat - target) <= C.quantization_error_bound(q01, q99)[None, :] + 1e-9)

    # the label really is the next observed state, which is what makes A_t causal
    t = D.valid_frames(len(ep["state"]))[:64]
    assert np.allclose(ep["action"][t], ep["state"][t + 1])
