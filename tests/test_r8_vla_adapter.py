#!/usr/bin/env python3
"""R8-B tests: 42D adapter, velocity-residual command path, metadata, projector width.

CPU only. The 7B forward/backward is covered by diagnostics/r8_smoke.py (GPU), which is a
separate gate; these tests must never be presented as proof that the VLA integration works.
"""
import json
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("code/xpolicylab/policy/OpenVLA_Pure", "diagnostics", "training"):
    sys.path.insert(0, os.path.join(REPO, sub))
_OFT = os.environ.get("PUSHT_OFT",
                      "/home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_OFT/openvla_oft")
for _p in (_OFT, os.path.join(_OFT, "vla-scripts")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import r8_vla_adapter as A                 # noqa: E402
import r8_contract as C                    # noqa: E402
import r8_normalization as N               # noqa: E402
import action_codec as AC                  # noqa: E402
import r6_data as D                        # noqa: E402

HDF5 = os.environ.get("PUSHT_HDF5", "/home/T7/ojh/robot_actual/data/data/hdf5/push_T/arx_x5/data")
SPLIT = os.environ.get("PUSHT_SPLIT", "/home/dgut/pusht_runtime/split80.json")
BOUNDS = "/home/dgut/pusht_runtime/push_T_bounds.npz"
CONTRACT_JSON = os.path.join(REPO, "evidence/r8_vla_42d_smoke_20261020/frozen_r8_contract.json")

PMIN = np.linspace(-2.0, -0.5, 14).astype(np.float32)
PMAX = np.linspace(2.0, 0.5, 14).astype(np.float32)


def contract():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.05, (400, 14)).astype(np.float32)
    a = rng.normal(0, 0.02, (400, 14)).astype(np.float32)
    return N.NormalizationContract.fit(v, v, a, PMIN, PMAX, N.SIGNED_MOTION, 0.99)


def episode(n=60, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.01, (n, 14)).astype(np.float32)
    S = np.cumsum(v, axis=0).astype(np.float32)
    Aa = np.vstack([S[1:], S[-1:]]).astype(np.float32)
    return S, Aa


class TestFeatureLayout(unittest.TestCase):
    def test_shape_and_boundary_rules(self):
        c = contract()
        S, _ = episode()
        p = A.build_proprio(S[5], S[4], S[3], c)
        self.assertEqual(p.shape, (42,))
        v, acc = A.history_terms(S[5], S[4], S[3])
        np.testing.assert_allclose(v, S[5] - S[4], atol=1e-6)
        np.testing.assert_allclose(acc, S[5] - 2 * S[4] + S[3], atol=1e-6)
        np.testing.assert_allclose(p[14:28], np.clip(v / c.v_scale, -1, 1), atol=1e-6)
        np.testing.assert_allclose(p[28:], np.clip(acc / c.a_scale, -1, 1), atol=1e-6)

    def test_frame0_and_frame1_boundaries(self):
        c = contract()
        S, _ = episode()
        buf = A.ProprioHistory42(c)
        f0, info0 = buf.observe(S[0])
        self.assertFalse(info0["velocity_defined"])
        np.testing.assert_allclose(f0[14:], 0.0, atol=1e-7)
        f1, info1 = buf.observe(S[1])
        self.assertTrue(info1["velocity_defined"])
        self.assertFalse(info1["acceleration_defined"])
        np.testing.assert_allclose(f1[28:], 0.0, atol=1e-7)
        f2, info2 = buf.observe(S[2])
        self.assertTrue(info2["acceleration_defined"])
        self.assertGreater(np.abs(f2[28:]).max(), 0.0)

    def test_online_equals_offline_and_env_isolation(self):
        c = contract()
        S, _ = episode()
        buf = A.ProprioHistory42(c)
        for t in range(4):
            online, _ = buf.observe(S[t])
            offline = (c.transform(S[0], np.zeros(14, np.float32), np.zeros(14, np.float32)) if t == 0
                       else A.build_proprio(S[t], S[t - 1], None, c) if t == 1
                       else A.build_proprio(S[t], S[t - 1], S[t - 2], c))
            np.testing.assert_allclose(online, offline, atol=1e-5)
        other, info = buf.observe(S[10], env_idx=1)
        self.assertFalse(info["velocity_defined"])
        buf.reset()
        again, info = buf.observe(S[10])
        self.assertFalse(info["velocity_defined"])


class TestTargetAndCommand(unittest.TestCase):
    def test_residual_target_and_exact_command(self):
        S, Aa = episode()
        v = S[7] - S[6]
        R = A.training_target(S[7], Aa[7], v)
        np.testing.assert_allclose(R, (Aa[7] - S[7]) - v, atol=1e-6)
        np.testing.assert_allclose(A.command_from_residual(S[7], v, R), Aa[7], atol=1e-6)

    def test_gripper_policy_and_cv_requirement(self):
        c = contract()
        S, Aa = episode()
        v = S[9] - S[8]
        R = A.training_target(S[9], Aa[9], v)
        q01, q99 = np.full(14, -0.05, np.float32), np.full(14, 0.05, np.float32)
        idx = AC.encode_indices(AC.normalize(R, q01, q99), mode=AC.MODE_FIXED)
        cmd, Rq, excluded = A.decode_tokens_to_command(idx, q01, q99, S[9], v, AC,
                                                      cv_command=S[9] + v)
        self.assertEqual(excluded, [6, 13])
        cv = S[9] + v
        np.testing.assert_allclose(cmd[[6, 13]], cv[[6, 13]], atol=1e-6)
        with self.assertRaises(ValueError):
            A.decode_tokens_to_command(idx, q01, q99, S[9], v, AC)

    def test_codec_bound(self):
        q01, q99 = np.full(14, -0.05, np.float32), np.full(14, 0.05, np.float32)
        rng = np.random.default_rng(2)
        R = rng.uniform(-0.03, 0.03, (256, 14)).astype(np.float32)
        idx = AC.encode_indices(AC.normalize(R, q01, q99), mode=AC.MODE_FIXED)
        dec = AC.denormalize(AC.decode_indices(idx, mode=AC.MODE_FIXED), q01, q99)
        step = (q99 - q01) / 255.0
        self.assertTrue(np.all(np.abs(dec - R) <= step + 1e-6))

    def test_clip_to_bounds(self):
        c = np.full(14, 5.0, np.float32)
        lo, hi = np.full(14, -1.0, np.float32), np.full(14, 1.0, np.float32)
        np.testing.assert_allclose(A.clip_command(c, lo, hi), hi)


class TestMetadata(unittest.TestCase):
    def test_roundtrip_and_fail_closed(self):
        c = contract()
        q01, q99 = np.full(14, -0.02, np.float32), np.full(14, 0.02, np.float32)
        meta = A.metadata(c, q01, q99)
        resolved = A.resolve_metadata(meta, projector_dim=42)
        self.assertEqual(resolved["contract"].mode, N.SIGNED_MOTION)
        np.testing.assert_allclose(resolved["q01"], q01)
        for broken in ({"proprio_mode": "svacc"},
                       dict(meta, proprio_dim=28),
                       dict(meta, target_representation="residual"),
                       {k: v for k, v in meta.items() if k != "q01"}):
            with self.assertRaises(ValueError):
                A.resolve_metadata(broken, projector_dim=42)

    @unittest.skipUnless(os.path.isfile(CONTRACT_JSON), "frozen contract absent")
    def test_frozen_contract_is_consistent(self):
        with open(CONTRACT_JSON) as fh:
            meta = json.load(fh)
        self.assertEqual(meta["normalization_mode"], "signed_motion")
        self.assertEqual(meta["proprio_dim"], 42)
        self.assertEqual(meta["target_representation"], "velocity_residual")
        self.assertEqual(meta["gripper_policy"]["gripper_dims"], [6, 13])
        r = A.resolve_metadata(meta, projector_dim=42)
        self.assertEqual(len(r["q01"]), 14)


class TestProjectorClass(unittest.TestCase):
    def test_real_projector_width_and_params(self):
        try:
            import torch
            from prismatic.models.projectors import ProprioProjector
        except Exception as exc:      # pragma: no cover
            self.skipTest("torch/prismatic unavailable: %s" % exc)
        proj = ProprioProjector(proprio_dim=42, llm_dim=128)
        x = torch.zeros(2, 42)
        y = proj(x)
        self.assertEqual(tuple(y.shape), (2, 128))
        self.assertEqual(int(proj.proprio_dim), 42)
        n = sum(p.numel() for p in proj.parameters())
        self.assertEqual(n, A.projector_param_count(128))

    def test_wrong_width_state_dict_rejected(self):
        try:
            import torch
            from prismatic.models.projectors import ProprioProjector
        except Exception as exc:      # pragma: no cover
            self.skipTest("torch/prismatic unavailable: %s" % exc)
        proj = ProprioProjector(proprio_dim=42, llm_dim=64)
        sd = {"fc1.weight": torch.zeros(64, 14), "fc1.bias": torch.zeros(64),
              "fc2.weight": torch.zeros(64, 64), "fc2.bias": torch.zeros(64)}
        with self.assertRaises(ValueError):
            A.load_projector_state(proj, sd)


@unittest.skipUnless(os.path.isdir(HDF5) and os.path.isfile(SPLIT), "real HDF5 unavailable")
class TestRealData(unittest.TestCase):
    def test_real_frames_no_leak_and_finite(self):
        c = contract()
        split = D.load_split(SPLIT)
        ep = D.load_episode(split["val"][0], HDF5)
        t = 50
        p = A.build_proprio(ep["state"][t], ep["state"][t - 1], ep["state"][t - 2], c)
        self.assertTrue(np.isfinite(p).all() and p.shape == (42,))
        S2 = ep["state"].copy(); S2[t + 1:] += 3.0
        p2 = A.build_proprio(S2[t], S2[t - 1], S2[t - 2], c)
        np.testing.assert_array_equal(p, p2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
