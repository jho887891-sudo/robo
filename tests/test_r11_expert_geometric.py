#!/usr/bin/env python3
"""Lane D RED tests -- scripted geometric Push-T expert (pure CPU, no simulator).

Nothing here touches torch, CUDA, isaacsim or RoboDojo: the expert is plain numpy and the
"simulator" used by the closed-loop test is a 20-line planar kinematic mock defined below.

Two contracts are pinned:

  1 GEOMETRY  -- the push direction comes from block->target, the pre-contact standoff sits
                 *behind* the block along -u, every push step advances <= 0.01 m along +u,
                 the phase machine is approach -> contact -> push -> retract -> done, and a
                 finished episode returns None instead of an invented action;
  2 HONESTY   -- no all-zero / random / constant 14D output, no silent zeros on missing or
                 non-finite input (raise instead), and a joint mapping that is missing or
                 degenerate raises rather than being papered over.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "training" / "r11_expert_geometric.py"

PHASES = ("approach", "contact", "push", "retract", "done")
HOME14 = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.50,
                   0.30, -0.10, 0.20, -0.30, 0.40, -0.50, 0.40], float)


def load():
    assert MODULE.is_file(), "missing expert module: %s" % MODULE
    spec = importlib.util.spec_from_file_location("r11_expert_geometric", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ScaleIK:
    """Transparent stand-in for the environment IK.

    joints[0:3] are a monotone scalar multiple of the commanded cartesian position, so a
    direction check in joint space is a direction check in cartesian space.  This proves the
    *waypoint logic*; real-robot joint feasibility still needs the simulator IK (see report).
    """

    def __init__(self, scale=10.0):
        self.scale = float(scale)
        self.calls = 0

    def __call__(self, pose7, arm="left"):
        p = np.asarray(pose7, float).reshape(-1)
        self.calls += 1
        return np.array([p[0] * self.scale, p[1] * self.scale, p[2] * self.scale * 0.5,
                         0.11, -0.22, 0.33, 0.50], float)


def quat_yaw_deg(deg):
    half = np.deg2rad(deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def pose7(xy, z=0.02, yaw_deg=0.0):
    return np.array([float(xy[0]), float(xy[1]), float(z)], float).tolist() + quat_yaw_deg(yaw_deg).tolist()


def expert(**kw):
    M = load()
    kw.setdefault("home_joints", HOME14)
    kw.setdefault("ik", ScaleIK())
    return M.GeometricPushTExpert(**kw)


def closed_loop(block_xy=(0.30, 0.10), target_xy=(0.36, 0.10), ee_xy=(0.10, 0.00),
                z=0.02, steps=400, contact_radius=0.005, **kw):
    """Planar kinematic mock of a rigid push.

    The end effector follows the commanded xy exactly; a contact carries the block along the
    commanded displacement only when the motion is directed *into* the block (so backing off
    separates instead of dragging).  This is deliberately kinematic: it validates the expert's
    waypoints, not the physics of the real scene.
    """
    E = expert(**kw)
    block = pose7(block_xy, z)
    target = pose7(target_xy, z)
    ee = pose7(ee_xy, z)
    u = (np.asarray(target_xy, float) - np.asarray(block_xy, float))
    u = u / np.linalg.norm(u)
    trace = []
    for step in range(steps):
        tgt14 = E.plan(block, target, ee, step)
        wp = None if tgt14 is None else np.asarray(E.last_waypoint, float)
        if tgt14 is None:
            assert wp is None, "plan() finished but still reported a waypoint"
            trace.append({"step": step, "phase": E.phase(), "wp": None, "a14": None,
                          "block": np.array(block[:2], float).copy(),
                          "ee": np.array(ee[:2], float).copy(), "u": u})
            break
        assert np.asarray(tgt14).shape == (14,)
        prev_ee = np.asarray(ee[:2], float).copy()
        # contact is a precondition of the step, not a consequence of arriving: the EE only
        # carries the block when it was already touching it before the commanded motion
        in_contact = (np.linalg.norm(prev_ee - np.asarray(block[:2], float))
                      <= contact_radius + 5e-4)
        trace.append({"step": step, "phase": E.phase(), "wp": np.asarray(wp, float),
                      "a14": np.asarray(tgt14, float), "u": u,
                      "goal": np.asarray(E.last_goal, float),
                      "block": np.array(block[:2], float).copy(),
                      "ee": prev_ee.copy()})
        ee = pose7(np.asarray(wp)[:2], z)
        d = np.asarray(ee[:2], float) - prev_ee
        toward = np.asarray(block[:2], float) - prev_ee
        if in_contact and float(np.dot(d, toward)) > 0.0:
            block = pose7(np.asarray(block[:2], float) + d, z)
    return E, trace, block, target


# ---------------------------------------------------------------- 1. geometry

def test_starts_in_approach_and_reports_a_legal_phase():
    E = expert()
    assert E.phase() == "approach"
    E.plan(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)
    assert E.phase() in PHASES


def test_standoff_waypoint_sits_behind_the_block_along_minus_push_direction():
    E = expert(standoff=0.02)
    block, target, ee = pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.00))
    wp = np.asarray(E.plan_cartesian(block, target, ee, 0), float)
    assert wp.shape == (7,)
    u = np.array([1.0, 0.0])                      # block -> target
    goal = np.asarray(E.last_goal, float)
    behind = goal[:2] - np.asarray(block[:2], float)
    assert float(np.dot(behind, u)) < 0.0, "approach point must be on the -u side of the block"
    assert float(np.dot(behind, u)) == pytest.approx(-0.02, abs=1e-9)
    assert abs(float(behind[1])) < 1e-9, "no lateral excursion on a pure +x push"
    assert wp[2] == pytest.approx(0.02), "planar expert works at block height"
    # a single call never jumps further than approach_step, and it heads straight for the goal
    step_vec = np.asarray(wp[:2], float) - np.asarray(ee[:2], float)
    assert np.linalg.norm(step_vec) <= 0.02 + 1e-9
    assert np.cross(step_vec, goal[:2] - np.asarray(ee[:2], float)) == pytest.approx(0.0, abs=1e-12)


def test_approach_rate_is_bounded_and_monotone_towards_the_standoff():
    E = expert(approach_step=0.02)
    block, target = pose7((0.30, 0.10)), pose7((0.36, 0.10))
    ee_xy = np.array([0.10, 0.00])
    standoff = np.array([0.28, 0.10])
    closing = []
    for step in range(40):
        wp = np.asarray(E.plan_cartesian(block, target, pose7(ee_xy), step), float)
        if E.phase() != "approach":
            break                       # this call already belonged to the contact phase
        assert np.linalg.norm(np.asarray(wp[:2], float) - ee_xy) <= 0.02 + 1e-9
        closing.append(float(np.linalg.norm(np.asarray(wp[:2], float) - standoff)))
        ee_xy = np.asarray(wp[:2], float)
    assert len(closing) >= 3, "the standoff is 206 mm away, so several approach steps are needed"
    assert all(b <= a + 1e-12 for a, b in zip(closing, closing[1:])), \
        "approach must close the distance to the standoff monotonically"
    assert closing[-1] <= 1e-9, "approach must land exactly on the standoff point"
    assert E.phase() == "contact"
    assert np.linalg.norm(ee_xy - standoff) <= 1e-9


def test_push_steps_advance_along_the_push_direction_and_never_exceed_10mm():
    E, trace, block, target = closed_loop()
    pushing = [r for r in trace if r["phase"] == "push"]
    assert len(pushing) >= 3, "target is 60 mm away, so several push steps are required"
    for r in pushing:
        d = np.asarray(r["wp"][:2], float) - r["ee"]
        assert 0.0 < np.linalg.norm(d) <= 0.01 + 1e-9
        assert float(np.dot(d, r["u"])) > 0.0, "push must move towards the target"
        assert abs(float(np.cross(r["u"], d))) < 1e-9, "push must stay on the push axis"


def test_diagonal_push_direction_is_normalised_and_correct():
    E = expert()
    block, target, ee = pose7((0.20, 0.20)), pose7((0.26, 0.26)), pose7((0.00, 0.00))
    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    first_push = None
    for step in range(80):
        wp = np.asarray(E.plan_cartesian(block, target, ee, step), float)
        if E.phase() == "push":
            first_push = wp
            break
        behind = wp[:2] - np.asarray(block[:2], float)
        assert float(np.dot(behind, u)) <= 1e-9, "the EE must stay on the -u side before pushing"
        ee = pose7(wp[:2])
    assert E.phase() == "push" and first_push is not None
    # the first push step runs along the diagonal and is bounded by 10 mm
    d = first_push[:2] - np.asarray(ee[:2], float)
    assert 0.0 < np.linalg.norm(d) <= 0.01 + 1e-9
    assert np.cross(u, d) == pytest.approx(0.0, abs=1e-12)


def test_closed_loop_reaches_the_target_retracts_and_finishes_with_none():
    E, trace, block, target = closed_loop()
    phases = [r["phase"] for r in trace]
    for name in ("approach", "contact", "push", "retract"):
        assert name in phases, "phase %r never happened: %r" % (name, phases)
    first = {p: phases.index(p) for p in ("approach", "contact", "push", "retract")}
    assert first["approach"] < first["contact"] < first["push"] < first["retract"]
    assert phases[-1] == "done"
    xy_err = float(np.linalg.norm(np.asarray(block[:2], float) - np.asarray(target[:2], float)))
    assert xy_err <= 0.007, "expert failed to push the block onto the pad (err=%.5f)" % xy_err
    assert E.phase() == "done"
    assert E.stop_reason == "reached"
    assert E.plan(block, target, pose7((0.0, 0.0)), len(trace)) is None
    # official success conjunction (lift 0, arms home)
    assert E.is_success(block, target, 0.0, True) is True


def test_retract_clears_the_block_before_heading_home():
    E, trace, block, target = closed_loop()
    retract = [r for r in trace if r["phase"] == "retract"]
    assert retract, "no retract step"
    # the first retract motions move away from the block along -u, not through it
    first = retract[0]
    d = np.asarray(first["wp"][:2], float) - first["ee"]
    assert float(np.dot(d, first["u"])) < 0.0
    last = retract[-1]
    assert float(np.linalg.norm(np.asarray(last["wp"][:2], float))) <= 0.05


def test_timeout_stops_the_episode_without_inventing_an_action():
    E = expert(max_steps=5)
    block, target, ee = pose7((0.30, 0.10)), pose7((0.60, 0.10)), pose7((0.10, 0.0))
    outs = []
    for step in range(8):
        outs.append(E.plan(block, target, ee, step))
    assert outs[-1] is None
    assert E.phase() == "done"
    assert E.stop_reason == "timeout"


def test_reset_restores_the_initial_state():
    E, trace, block, target = closed_loop(steps=60)
    assert E.phase() != "approach"
    E.reset()
    assert E.phase() == "approach"
    assert E.stop_reason == ""
    wp = E.plan_cartesian(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)
    assert wp is not None


# ---------------------------------------------------------------- 2. 14D output contract

def test_14d_targets_are_finite_nonzero_and_the_active_arm_tracks_the_push():
    E, trace, block, target = closed_loop()
    rows = [r["a14"] for r in trace if r["a14"] is not None]
    assert rows, "no 14D target was produced"
    for a in rows:
        assert a.shape == (14,)
        assert np.isfinite(a).all()
        assert np.abs(a).max() > 0.0, "all-zero 14D action is forbidden"
    pushing = [r for r in trace if r["phase"] == "push"]
    assert len(pushing) >= 3
    # joint space direction of the active arm must follow the cartesian push direction
    for prev, cur in zip(pushing, pushing[1:]):
        delta = np.asarray(cur["a14"][0:2], float) - np.asarray(prev["a14"][0:2], float)
        assert float(np.dot(delta, cur["u"])) > 0.0
        assert abs(float(np.cross(cur["u"], delta))) < 1e-9


def test_passive_arm_and_grippers_hold_a_documented_home_pose():
    E, trace, block, target = closed_loop()
    a = np.asarray(trace[0]["a14"], float)
    np.testing.assert_allclose(a[7:14], HOME14[7:14], atol=0.0)
    np.testing.assert_allclose(a[6], HOME14[6], atol=0.0)
    for r in trace:
        if r["a14"] is None:
            continue
        np.testing.assert_allclose(np.asarray(r["a14"])[6:7], HOME14[6:7], atol=1e-12)


def test_two_identical_calls_produce_identical_targets():
    E = expert()
    block, target, ee = pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0))
    a = np.asarray(E.plan(block, target, ee, 0), float)
    E.reset()
    b = np.asarray(E.plan(block, target, ee, 0), float)
    np.testing.assert_array_equal(a, b)


def test_missing_ik_raises_instead_of_returning_a_placeholder():
    M = load()
    E = M.GeometricPushTExpert(home_joints=HOME14)
    with pytest.raises(Exception) as exc:
        E.plan(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)
    assert "ik" in str(exc.value).lower()


def test_missing_home_joints_raises():
    M = load()
    E = M.GeometricPushTExpert(ik=ScaleIK())
    with pytest.raises(Exception) as exc:
        E.plan(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)
    assert "home" in str(exc.value).lower()


def test_constant_ik_is_rejected_as_a_fake_expert():
    M = load()
    E = M.GeometricPushTExpert(home_joints=HOME14, ik=lambda p, arm="left": np.zeros(7))
    with pytest.raises(Exception):
        E.plan(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)


def test_ik_returning_a_nonfinite_joint_vector_is_rejected():
    M = load()
    bad = np.array([np.nan, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5])
    E = M.GeometricPushTExpert(home_joints=HOME14, ik=lambda p, arm="left": bad)
    with pytest.raises(Exception):
        E.plan(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)


def test_ik_returning_the_wrong_shape_is_rejected():
    M = load()
    E = M.GeometricPushTExpert(home_joints=HOME14, ik=lambda p, arm="left": np.zeros(6))
    with pytest.raises(Exception):
        E.plan(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)


# ---------------------------------------------------------------- 3. fail-closed inputs

@pytest.mark.parametrize("bad", [None, float("nan"), "0.3,0.1", np.zeros(1), np.zeros(0)])
def test_missing_or_malformed_pose_raises(bad):
    E = expert()
    block, target, ee = pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0))
    for args in ((bad, target, ee), (block, bad, ee), (block, target, bad)):
        with pytest.raises(Exception):
            E.plan(*args, 0)


@pytest.mark.parametrize("field", [0, 1, 2])
def test_nonfinite_pose_component_raises(field):
    E = expert()
    bad = pose7((0.30, 0.10))
    bad[field] = float("inf")
    with pytest.raises(Exception):
        E.plan(bad, pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)
    with pytest.raises(Exception):
        E.plan_cartesian(bad, pose7((0.36, 0.10)), pose7((0.10, 0.0)), 0)


def test_undefined_push_direction_raises():
    E = expert(xy_tol=1e-15)
    block = pose7((0.30, 0.10))
    target = pose7((0.30 + 1e-12, 0.10))
    with pytest.raises(Exception):
        E.plan(block, target, pose7((0.10, 0.0)), 0)


def test_negative_step_raises():
    E = expert()
    with pytest.raises(Exception):
        E.plan(pose7((0.30, 0.10)), pose7((0.36, 0.10)), pose7((0.10, 0.0)), -1)


def test_bad_constructor_arguments_raise():
    M = load()
    for kw in ({"xy_tol": -1.0}, {"standoff": 0.0}, {"max_steps": 0}, {"push_step": 0.05},
               {"z_max": -0.1}, {"active_arm": "middle"}):
        with pytest.raises(Exception):
            M.GeometricPushTExpert(**kw)


# ---------------------------------------------------------------- 4. official success rule

def test_is_success_follows_the_official_conjunction():
    E = expert()
    blk = pose7((0.30, 0.10))
    tgt = pose7((0.30, 0.10))
    assert E.is_success(blk, tgt, 0.0, True) is True
    assert E.is_success(pose7((0.3069, 0.10)), tgt, 0.0, True) is True       # 6.9 mm
    assert E.is_success(pose7((0.3071, 0.10)), tgt, 0.0, True) is False      # 7.1 mm
    assert E.is_success(blk, tgt, 0.009, True) is True                       # lift under 10 mm
    assert E.is_success(blk, tgt, 0.011, True) is False                      # block was lifted
    assert E.is_success(blk, tgt, 0.0, False) is False                       # arms not home


def test_is_success_checks_orientation_when_quaternions_are_present():
    E = expert()
    tgt = pose7((0.30, 0.10), yaw_deg=0.0)
    assert E.is_success(pose7((0.30, 0.10), yaw_deg=5.0), tgt, 0.0, True) is True
    assert E.is_success(pose7((0.30, 0.10), yaw_deg=10.0), tgt, 0.0, True) is False


def test_is_success_rejects_bad_arguments():
    E = expert()
    with pytest.raises(Exception):
        E.is_success(None, pose7((0.30, 0.10)), 0.0, True)
    with pytest.raises(Exception):
        E.is_success(pose7((0.30, 0.10)), pose7((0.30, 0.10)), float("nan"), True)
    with pytest.raises(Exception):
        E.is_success(pose7((0.30, 0.10)), pose7((0.30, 0.10)), 0.0, "yes")


def test_success_inside_the_writer_attainable_window():
    """The gate must be strict enough to be a real gate but loose enough to be reachable."""
    E = expert()
    for mm in (0.0, 3.0, 6.9):
        p = pose7((0.30 + mm / 1000.0, 0.10))
        assert E.is_success(p, pose7((0.30, 0.10)), 0.0, True) is True
    for mm in (7.1, 12.0):
        p = pose7((0.30 + mm / 1000.0, 0.10))
        assert E.is_success(p, pose7((0.30, 0.10)), 0.0, True) is False
