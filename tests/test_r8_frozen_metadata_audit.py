#!/usr/bin/env python3
"""CPU regressions: frozen R8 target quantiles must not use DEV and decoder must honor them."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load_as(name, rel):
    spec=importlib.util.spec_from_file_location(name,str(ROOT/rel))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

F=load_as("r8_frozen_contract_builder","tools/make_frozen_contract.py")
S=load_as("r8_smoke_contract_audit","diagnostics/r8_smoke.py")

class TestFrozenR8QuantileProvenance(unittest.TestCase):
    def setUp(self):
        self.split={"train":["e%03d"%i for i in range(80)],
                    "val":["e%03d"%i for i in range(80,100)]}

    def test_64_train_16_dev_never_80(self):
        eps,dev=F.select_quantile_train_episodes(self.split)
        self.assertEqual((len(eps),len(dev)),(64,16))
        self.assertFalse(set(eps)&set(dev))
        self.assertFalse(set(eps)&set(self.split["val"]))
        self.assertFalse(set(dev)&set(self.split["val"]))

    def test_limit_cannot_reintroduce_dev(self):
        eps,dev=F.select_quantile_train_episodes(self.split,20)
        self.assertEqual(len(eps),20)
        with self.assertRaises(ValueError):
            F.select_quantile_train_episodes(self.split,80)

    def good_meta(self):
        return {"q01":[-0.01]*14,"q99":[0.01]*14,
                "frozen_from":{"n_train_episodes_for_quantiles":64,
                               "n_dev_episodes_excluded":16}}

    def test_quantiles_are_taken_from_metadata(self):
        a,b=S.frozen_quantiles(self.good_meta())
        np.testing.assert_array_equal(a,np.full(14,-0.01,np.float32))
        np.testing.assert_array_equal(b,np.full(14,0.01,np.float32))

    def test_80_episode_quantiles_fail_closed(self):
        meta=self.good_meta()
        meta["frozen_from"]["n_train_episodes_for_quantiles"]=80
        with self.assertRaises(ValueError): S.frozen_quantiles(meta)

    def test_missing_provenance_and_degenerate_bins_fail(self):
        meta=self.good_meta();meta.pop("frozen_from")
        with self.assertRaises(ValueError): S.frozen_quantiles(meta)
        meta=self.good_meta();meta["q99"]=[-0.02]*14
        with self.assertRaises(ValueError): S.frozen_quantiles(meta)

    def test_kv_harness_uses_shared_prefix(self):
        import inspect
        src=inspect.getsource(S.kv_diagnose)
        self.assertIn("W0 + tf",src)
        self.assertNotIn("W0 + tk",src)
        self.assertIn("same_prefix_teacher_forced",src)
        self.assertNotIn("numerical sensitivity of bf16 argmax at near-ties",src)

if __name__=="__main__":
    unittest.main(verbosity=2)
