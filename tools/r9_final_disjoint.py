#!/usr/bin/env python3
"""R9 FINAL-set provenance gate: compare episode CONTENT, not only filenames.

CPU only. Fail closed before freeze. This does not download or evaluate data.
"""
import argparse
import hashlib
import json
import os
import sys


def check(candidate, used_manifest, split, protocol):
    episodes = candidate.get('episodes', [])
    used = used_manifest.get('episodes', [])
    used_names = {r['file'] for r in used} | set(split['train']) | set(split['val'])
    used_hashes = {r['content_sha256'] for r in used}
    used_state_action_hashes = {r['state_action_sha256'] for r in used}
    names = [r['file'] for r in episodes]
    hashes = [r['content_sha256'] for r in episodes]
    state_action_hashes = [r['state_action_sha256'] for r in episodes]
    if not episodes or not used or not used_hashes:
        raise ValueError('candidate and complete used-reference manifests are required')
    all_used_names = set(split['train']) | set(split['val'])
    if {r['file'] for r in used} != all_used_names or len(used) != len(all_used_names):
        raise ValueError('used-reference manifest names differ from frozen split')
    if len(used_hashes) != len(used) or len(used_state_action_hashes) != len(used):
        raise ValueError('used-reference manifest contains duplicate trajectory content')
    expected_dataset_hash = protocol['exclusion']['must_be_content_disjoint_from']['dataset_sha256_of_used_100']
    calculated = hashlib.sha256(''.join(sorted(used_hashes)).encode()).hexdigest()
    if calculated != expected_dataset_hash or used_manifest.get('dataset_sha256') != expected_dataset_hash:
        raise ValueError('used-reference manifest identity does not match the frozen 100-episode dataset')
    min_final = int(protocol['splits']['final_episodes_min'])
    n_dev = int(protocol['splits']['dev_episodes'])
    req = protocol['inclusion']
    overlap_names = sorted(set(names) & used_names)
    overlap_content = sorted(set(hashes) & used_hashes)
    overlap_state_action = sorted(set(state_action_hashes) & used_state_action_hashes)
    dup_names = len(names) - len(set(names))
    dup_content = len(hashes) - len(set(hashes))
    dup_state_action = len(state_action_hashes) - len(set(state_action_hashes))
    invalid = [r['file'] for r in episodes if
               req['instruction_contains'].lower() not in r.get('instruction', '').lower()
               or r.get('frequency') != req['frequency_hz']
               or int(r.get('length', 0)) < req['min_frames']]
    n_final = len(episodes) - n_dev
    # The reference manifest is itself checked against the full 100-episode split.
    missing_used_names = sorted((set(split['train']) | set(split['val'])) - {r['file'] for r in used})
    ok = not (overlap_names or overlap_content or overlap_state_action
              or dup_names or dup_content or dup_state_action or invalid
              or missing_used_names) and n_final >= min_final
    return {'pass': bool(ok), 'new_episodes': len(episodes), 'dev_episodes': n_dev,
            'final_episodes': n_final, 'min_final_episodes': min_final,
            'name_overlap': overlap_names, 'content_overlap_sha256': overlap_content,
            'state_action_overlap_sha256': overlap_state_action,
            'duplicate_names': dup_names, 'duplicate_content': dup_content,
            'duplicate_state_action': dup_state_action,
            'invalid_task_or_frequency_or_length': invalid,
            'missing_used_reference_names': missing_used_names,
            'note': 'Gate g3/g4 only; still require real HDF5 camera/shape validation, verify, freeze and commit before evaluation'}


def main():
    p = argparse.ArgumentParser()
    for key in ('candidate', 'used_manifest', 'split', 'protocol', 'out'):
        p.add_argument('--' + key.replace('_', '-'), required=True)
    a = p.parse_args()
    with open(a.candidate) as f: candidate = json.load(f)
    with open(a.used_manifest) as f: used = json.load(f)
    with open(a.split) as f: split = json.load(f)
    with open(a.protocol) as f: protocol = json.load(f)
    result = check(candidate, used, split, protocol)
    # Do not create directories, or replace an existing result silently.
    if os.path.exists(a.out):
        raise FileExistsError('output already exists: ' + a.out)
    with open(a.out, 'x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result['pass'] else 4


if __name__ == '__main__':
    sys.exit(main())
