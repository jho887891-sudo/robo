#!/usr/bin/env python3
"""CPU synthetic regression checks for signed motion normalization information."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

p=Path(__file__).resolve().parents[1]/"diagnostics/r8_scale_audit.py"
spec=importlib.util.spec_from_file_location("r8_scale_audit",p)
M=importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class R8MotionScaleAudit(unittest.TestCase):
    def test_nonnegative_position_range_collapses_negative_and_zero_velocity(self):
        lo=np.zeros(14,np.float32)
        hi=np.ones(14,np.float32)
        v=np.array([-0.01,0.0,0.01],np.float32)[:,None]*np.ones((1,14),np.float32)
        old=M.original_transform(v,lo,hi)
        np.testing.assert_array_equal(old[:2],np.full((2,14),-1.,np.float32))
        self.assertTrue((old[2]>-1).all())
        alt=M.signed_transform(v,np.full(14,0.1,np.float32))
        self.assertLess(alt[0,0],0)
        self.assertEqual(alt[1,0],0)
        self.assertGreater(alt[2,0],0)

    def test_zero_velocity_stays_zero_in_signed_normalization(self):
        z=np.zeros((3,14),np.float32)
        np.testing.assert_array_equal(M.signed_transform(z,np.ones(14)),z)

    def test_train_fit_does_not_depend_on_future_or_holdout(self):
        train=np.vstack([np.full((125,14),0.001),np.full((125,14),-0.002)]).astype(np.float32)
        scale=M.train_only_scale(train)
        np.testing.assert_allclose(scale,0.002,rtol=1e-6)
        np.testing.assert_array_equal(scale,M.train_only_scale(train))
        holdout=np.full((10,14),999,np.float32)
        self.assertTrue((np.abs(M.signed_transform(holdout,scale))<=1).all())

    def test_invalid_training_data_fails(self):
        with self.assertRaises(ValueError):
            M.train_only_scale(np.zeros((10,14),np.float32))

    def test_history_arrays_use_only_past_frames(self):
        s=np.arange(140,dtype=np.float32).reshape(10,14)
        v,a=M.history_arrays([{"state":s}])
        self.assertEqual(v.shape,(7,14))
        np.testing.assert_array_equal(v,np.full_like(v,14))
        np.testing.assert_array_equal(a,np.zeros_like(a))


if __name__=="__main__":
    unittest.main(verbosity=2)
