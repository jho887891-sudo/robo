#!/usr/bin/env python3
"""R16 T2/T3: RoboDojo observation dict -> 42D inputs, and 14D command -> 4-key action dict.

CPU only, no model, no GPU, no simulator.  Every rejection here is a fail-closed case that would
otherwise corrupt a rollout silently (swapped cameras, swapped joints, an RGBA frame, a float
frame, a 25 Hz -> 30 Hz control-rate change, a gripper value the sim would clip).
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_R42"))
sys.path.insert(0, os.path.join(REPO, "code", "xpolicylab", "policy", "OpenVLA_Pure"))

import r42_adapter as R  # noqa: E402
import r42_env_io as E  # noqa: E402

# Distinct constants per joint so a reordering cannot pass unnoticed.
LEFT_ARM = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
LEFT_EE = [7.0]
RIGHT_ARM = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
RIGHT_EE = [14.0]
FLAT = np.asarray(LEFT_ARM + LEFT_EE + RIGHT_ARM + RIGHT_EE, np.float32)


def _rgb(shape=(4, 5, 3), seed=0):
    rs = np.random.RandomState(seed)
    return rs.randint(0, 255, shape).astype(np.uint8)


def _state_dict():
    return {"left_arm_joint_state": np.asarray(LEFT_ARM, np.float32),
            "left_ee_joint_state": np.asarray(LEFT_EE, np.float32),
            "right_arm_joint_state": np.asarray(RIGHT_ARM, np.float32),
            "right_ee_joint_state": np.asarray(RIGHT_EE, np.float32),
            "left_ee_pose": np.zeros(7, np.float32),
            "right_ee_pose": np.zeros(7, np.float32)}


def _obs(**over):
    """Exactly the dict the audited client sends (obs_manager.py:76-118 + eval_env.py:313-318)."""
    obs = {"vision": {cam: {"color": _rgb(seed=i)}
                      for i, cam in enumerate(E.CAMERA_ORDER)},
           "state": _state_dict(),
           "action": _state_dict(),
           "instruction": "Push the T-shaped block to align it precisely with the gray pad.",
           "additional_info": {"frequency": 25},
           "data_format_version": "v1.0",
           "env_idx": 0}
    obs.update(over)
    return obs


def _convert(**over):
    return E.obs_dict_to_inputs(_obs(**over), expected_hw=None)


# --------------------------------------------------------------------- T2: observation -> inputs
def test_client_observation_converts_to_the_frozen_contract():
    out = _convert()
    assert tuple(out["images"]) == E.CAMERA_ORDER
    assert all(v.dtype == np.uint8 for v in out["images"].values())
    assert out["state"].shape == (14,)
    assert out["state"].dtype == np.float32
    assert np.array_equal(out["state"], FLAT)
    assert out["env_idx"] == 0


def test_production_480x640_frames_are_accepted_at_the_default_contract():
    obs = _obs(vision={cam: {"color": _rgb((480, 640, 3), seed=i)}
                       for i, cam in enumerate(E.CAMERA_ORDER)})
    out = E.obs_dict_to_inputs(obs)                      # default expected_hw == (480, 640, 3)
    assert out["images"]["cam_head"].shape == (480, 640, 3)


def test_joint_order_is_left_arm_left_ee_right_arm_right_ee():
    out = _convert()
    assert list(out["state"][:6]) == LEFT_ARM
    assert list(out["state"][6:7]) == LEFT_EE
    assert list(out["state"][7:13]) == RIGHT_ARM
    assert list(out["state"][13:]) == RIGHT_EE
    assert E.GRIPPER_DIMS == tuple(R.GRIPPER_DIMS) == (6, 13)


def test_flat_state_and_dict_state_agree():
    flat = E.obs_dict_to_inputs(_obs(state=FLAT.copy()), expected_hw=None)
    assert np.array_equal(flat["state"], _convert()["state"])


def test_camera_dict_order_is_irrelevant_but_keys_are_not():
    seed_of = {"cam_head": 2, "cam_left_wrist": 1, "cam_right_wrist": 0}
    vision = {cam: {"color": _rgb(seed=seed_of[cam])} for cam in reversed(E.CAMERA_ORDER)}
    out = E.obs_dict_to_inputs(_obs(vision=vision), expected_hw=None)
    assert tuple(out["images"]) == E.CAMERA_ORDER
    for cam in E.CAMERA_ORDER:
        assert np.array_equal(out["images"][cam], _rgb(seed=seed_of[cam])), cam


def test_er_pose_keys_are_ignored_not_fused():
    out = _convert()
    assert out["state"].shape == (14,)
    assert "left_ee_pose" in E.STATE_IGNORED_KEYS


def test_recorded_action_is_dropped_and_reported():
    out = _convert()
    assert "action" in out["dropped"]
    assert "action" not in out
    assert E.assert_no_action_leak(out) is True


def test_unknown_camera_name_is_refused():
    vision = {cam: {"color": _rgb()} for cam in E.CAMERA_ORDER}
    vision["cam_side"] = {"color": _rgb()}
    with pytest.raises(E.R42EnvIOError, match="unexpected camera"):
        E.obs_dict_to_inputs(_obs(vision=vision), expected_hw=None)


def test_missing_camera_is_refused():
    vision = {cam: {"color": _rgb()} for cam in E.CAMERA_ORDER if cam != "cam_right_wrist"}
    with pytest.raises(E.R42EnvIOError, match="missing camera"):
        E.obs_dict_to_inputs(_obs(vision=vision), expected_hw=None)


def test_declared_camera_names_must_be_the_frozen_order():
    with pytest.raises(E.R42EnvIOError, match="camera_names"):
        E.obs_dict_to_inputs(_obs(camera_names=["cam_head", "cam_right_wrist",
                                                "cam_left_wrist"]), expected_hw=None)


@pytest.mark.parametrize("bad", [
    "rgba", "float", "chw", "two_d", "wrong_resolution", "empty",
])
def test_bad_image_layouts_fail_closed(bad):
    vision = {cam: {"color": _rgb()} for cam in E.CAMERA_ORDER}
    if bad == "rgba":
        vision["cam_head"] = {"color": _rgb((4, 5, 4))}
    elif bad == "float":
        vision["cam_head"] = {"color": _rgb().astype(np.float32) / 255.0}
    elif bad == "chw":
        vision["cam_head"] = {"color": np.zeros((3, 4, 5), np.uint8)}
    elif bad == "two_d":
        vision["cam_head"] = {"color": np.zeros((4, 5), np.uint8)}
    elif bad == "wrong_resolution":
        vision["cam_head"] = {"color": _rgb((240, 320, 3))}
    elif bad == "empty":
        vision["cam_head"] = {"color": np.zeros((0, 5, 3), np.uint8)}
    with pytest.raises(E.R42EnvIOError):
        E.obs_dict_to_inputs(_obs(vision=vision),
                             expected_hw=(480, 640, 3) if bad == "wrong_resolution" else None)


def test_shape_annotator_is_checked_not_trusted():
    vision = {cam: {"color": _rgb(), "shape": (4, 5, 3)} for cam in E.CAMERA_ORDER}
    assert E.obs_dict_to_inputs(_obs(vision=vision), expected_hw=None)["images"]
    vision["cam_head"] = {"color": _rgb(), "shape": (5, 4, 3)}
    with pytest.raises(E.R42EnvIOError, match="declares shape"):
        E.obs_dict_to_inputs(_obs(vision=vision), expected_hw=None)


def test_unknown_annotator_is_refused():
    vision = {cam: {"color": _rgb()} for cam in E.CAMERA_ORDER}
    vision["cam_head"]["depth"] = np.zeros((4, 5), np.float32)
    with pytest.raises(E.R42EnvIOError, match="annotator"):
        E.obs_dict_to_inputs(_obs(vision=vision), expected_hw=None)


def test_vision_missing_entirely_is_refused():
    obs = _obs()
    obs.pop("vision")
    with pytest.raises(E.R42EnvIOError, match="missing"):
        E.obs_dict_to_inputs(obs, expected_hw=None)


def test_state_joint_dim_must_match_exactly():
    state = _state_dict()
    state["left_arm_joint_state"] = np.zeros(5, np.float32)
    with pytest.raises(E.R42EnvIOError, match="left_arm_joint_state"):
        E.obs_dict_to_inputs(_obs(state=state), expected_hw=None)


def test_state_unknown_key_is_refused():
    state = _state_dict()
    state["left_ee_torque"] = np.zeros(1, np.float32)
    with pytest.raises(E.R42EnvIOError, match="unknown key"):
        E.obs_dict_to_inputs(_obs(state=state), expected_hw=None)


def test_state_missing_joint_key_is_refused():
    state = _state_dict()
    state.pop("right_ee_joint_state")
    with pytest.raises(E.R42EnvIOError, match="missing joint key"):
        E.obs_dict_to_inputs(_obs(state=state), expected_hw=None)


def test_non_finite_state_is_refused():
    state = _state_dict()
    state["left_arm_joint_state"] = np.asarray([1, 2, 3, 4, 5, np.nan], np.float32)
    with pytest.raises(E.R42EnvIOError, match="non-finite"):
        E.obs_dict_to_inputs(_obs(state=state), expected_hw=None)


def test_flat_state_of_the_wrong_length_is_refused():
    with pytest.raises(E.R42EnvIOError, match="14D"):
        E.obs_dict_to_inputs(_obs(state=np.zeros(13, np.float32)), expected_hw=None)


def test_control_rate_change_is_refused():
    with pytest.raises(E.R42EnvIOError, match="control rate"):
        E.obs_dict_to_inputs(_obs(additional_info={"frequency": 30}), expected_hw=None)


def test_unknown_additional_info_key_is_refused():
    with pytest.raises(E.R42EnvIOError, match="additional_info"):
        E.obs_dict_to_inputs(_obs(additional_info={"frequency": 25, "shm": 1}),
                             expected_hw=None)


def test_privileged_field_is_refused_before_anything_is_packed():
    with pytest.raises(R.R42Error, match="privileged|target"):
        E.obs_dict_to_inputs(_obs(target_t=np.zeros(3, np.float32)), expected_hw=None)


def test_unrecognised_top_level_key_is_refused():
    with pytest.raises(E.R42EnvIOError, match="unrecognised observation key"):
        E.obs_dict_to_inputs(_obs(debug_blob={"a": 1}), expected_hw=None)


def test_instruction_must_be_a_string():
    with pytest.raises(E.R42EnvIOError, match="instruction"):
        E.obs_dict_to_inputs(_obs(instruction=3), expected_hw=None)


def test_negative_env_idx_is_refused():
    with pytest.raises(E.R42EnvIOError, match="env_idx"):
        E.obs_dict_to_inputs(_obs(env_idx=-1), expected_hw=None)


def test_observation_must_be_a_dict():
    with pytest.raises(E.R42EnvIOError, match="dict"):
        E.obs_dict_to_inputs([1, 2, 3], expected_hw=None)


# ------------------------------------------------------------------ T3: 14D command -> action dict
def test_cmd14_becomes_the_four_key_action_dict():
    cmd = np.concatenate([np.asarray(LEFT_ARM), np.asarray([0.25]),
                          np.asarray(RIGHT_ARM), np.asarray([0.75])]).astype(np.float32)
    action = E.cmd14_to_action_dict(cmd)
    assert list(action) == ["left_arm_joint_state", "left_ee_joint_state",
                            "right_arm_joint_state", "right_ee_joint_state"]
    assert action["left_arm_joint_state"] == LEFT_ARM
    assert action["left_ee_joint_state"] == [0.25]
    assert action["right_arm_joint_state"] == RIGHT_ARM
    assert action["right_ee_joint_state"] == [0.75]
    assert all(isinstance(v, list) and all(isinstance(x, float) for x in v)
               for v in action.values())
    assert E.assert_action_dict(action) is True


def test_action_dict_roundtrips_to_the_same_14d_vector():
    cmd = np.asarray([0.1 * i - 0.5 for i in range(14)], np.float32)
    cmd[6] = 0.25
    cmd[13] = 0.75
    action = E.cmd14_to_action_dict(cmd)
    back = E.action_dict_to_cmd14(action)
    assert np.array_equal(back, cmd.astype(np.float32))


@pytest.mark.parametrize("bad", ["short", "long", "two_d", "nan", "gripper_high",
                                 "gripper_low", "text"])
def test_bad_commands_fail_closed(bad):
    cmd = np.zeros(14, np.float32)
    if bad == "short":
        cmd = np.zeros(13, np.float32)
    elif bad == "long":
        cmd = np.zeros(15, np.float32)
    elif bad == "two_d":
        cmd = np.zeros((1, 14), np.float32)
    elif bad == "nan":
        cmd[3] = np.nan
    elif bad == "gripper_high":
        cmd[6] = 1.5
    elif bad == "gripper_low":
        cmd[13] = -0.5
    elif bad == "text":
        cmd = np.asarray(["a"] * 14)
    with pytest.raises(E.R42EnvIOError):
        E.cmd14_to_action_dict(cmd)


def test_action_dict_missing_key_is_refused():
    action = E.cmd14_to_action_dict(np.zeros(14, np.float32))
    action.pop("right_ee_joint_state")
    with pytest.raises(E.R42EnvIOError, match="missing key"):
        E.action_dict_to_cmd14(action)


def test_action_dict_extra_key_is_refused():
    action = E.cmd14_to_action_dict(np.zeros(14, np.float32))
    action["left_ee_pose"] = [0.0] * 7
    with pytest.raises(E.R42EnvIOError, match="cannot consume"):
        E.action_dict_to_cmd14(action)


def test_action_dict_wrong_dim_is_refused():
    action = E.cmd14_to_action_dict(np.zeros(14, np.float32))
    action["left_arm_joint_state"] = [0.0] * 7
    with pytest.raises(E.R42EnvIOError, match="left_arm_joint_state"):
        E.action_dict_to_cmd14(action)


def test_action_dict_non_numeric_is_refused():
    action = E.cmd14_to_action_dict(np.zeros(14, np.float32))
    action["left_ee_joint_state"] = ["open"]
    with pytest.raises(E.R42EnvIOError):
        E.action_dict_to_cmd14(action)


def test_contract_declares_the_frozen_io():
    c = E.contract()
    assert c["camera_order"] == ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    assert c["joint_dims"] == [6, 1, 6, 1]
    assert c["state_dim"] == 14 and c["gripper_dims"] == [6, 13]
    assert c["control_hz"] == 25.0 and abs(c["control_dt"] - 0.04) < 1e-12
    assert c["image_dtype"] == "uint8" and c["expected_image_shape"] == [480, 640, 3]
