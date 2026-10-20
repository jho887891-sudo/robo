#!/usr/bin/env python3
"""Negative controls for R9 cross-dataset content-hash leakage."""
import os
import sys
import hashlib
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from r9_final_disjoint import check

REFERENCE_HASH = hashlib.sha256(''.join(sorted(['oldA', 'oldB'])).encode()).hexdigest()
P = {'exclusion': {'must_be_content_disjoint_from': {'dataset_sha256_of_used_100': REFERENCE_HASH}},
     'splits': {'dev_episodes': 5, 'final_episodes_min': 15},
     'inclusion': {'instruction_contains': 'Push the T-shaped block', 'frequency_hz': 25, 'min_frames': 100}}
S = {'train': ['used_a.hdf5'], 'val': ['used_b.hdf5']}
U = {'dataset_sha256': REFERENCE_HASH, 'episodes': [
       {'file': 'used_a.hdf5', 'content_sha256': 'oldA', 'state_action_sha256': 'oldStateA'},
       {'file': 'used_b.hdf5', 'content_sha256': 'oldB', 'state_action_sha256': 'oldStateB'}]}

def fresh():
    return {'episodes': [{'file': 'new_%d.hdf5' % i, 'content_sha256': 'fresh%d' % i, 'state_action_sha256': 'freshState%d' % i,
                          'instruction': 'Push the T-shaped block', 'frequency': 25, 'length': 200}
                         for i in range(20)]}

class Gate(unittest.TestCase):
    def test_fresh_passes(self):
        self.assertTrue(check(fresh(), U, S, P)['pass'])
    def test_renamed_old_content_fails(self):
        x = fresh(); x['episodes'][0]['content_sha256'] = 'oldA'
        r = check(x,U,S,P)
        self.assertFalse(r['pass']); self.assertEqual(r['content_overlap_sha256'], ['oldA'])
    def test_same_state_action_different_instruction_hash_fails(self):
        x = fresh(); x['episodes'][0]['state_action_sha256'] = 'oldStateA'
        self.assertFalse(check(x,U,S,P)['pass'])
    def test_used_reference_not_anchored_fails(self):
        x = fresh(); used = dict(U); used['dataset_sha256'] = 'wrong'
        with self.assertRaises(ValueError): check(x,used,S,P)
    def test_same_name_fails(self):
        x = fresh(); x['episodes'][0]['file'] = 'used_b.hdf5'
        self.assertFalse(check(x,U,S,P)['pass'])
    def test_duplicate_content_fails(self):
        x = fresh(); x['episodes'][1]['content_sha256'] = x['episodes'][0]['content_sha256']
        self.assertFalse(check(x,U,S,P)['pass'])
    def test_final_too_small_fails(self):
        x = fresh(); x['episodes'] = x['episodes'][:19]
        self.assertFalse(check(x,U,S,P)['pass'])
    def test_cross_task_fails(self):
        x = fresh(); x['episodes'][0]['instruction'] = 'Stack the three bowls'
        self.assertFalse(check(x,U,S,P)['pass'])
    def test_reference_incomplete_fails(self):
        x = fresh(); used = {'episodes': U['episodes'][:1]}
        with self.assertRaises(ValueError): check(x,used,S,P)

if __name__ == '__main__':
    unittest.main()
