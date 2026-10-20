# R10 evidence provenance (pinned copies)

The R10 CPU gates must be recomputable from this single branch, so a few small artefacts are
copied here byte-identically from their authoritative branches. Nothing is edited; the SHA256
below is the file as committed here.

| file | authoritative source | note |
|---|---|---|
| `evidence/r9_gate_20261020/frozen_contract_64train.json` | `experiment/r9-freeze-64train@f7b8b39` | 64-TRAIN frozen 42D/R contract (`2ab68fd3…`), used by the KV probe |
| `evidence/r9_eval_prep_20261020/manifest_existing100.json` | `research/r9-new-test-readiness@f8e43b0` | used-reference manifest for cross-set dedup (dataset `6757696…`) |
| `protocol/r9_eval_protocol.json` | `research/r9-new-test-readiness@f8e43b0` | pre-registered metrics/gates |
| `protocol/r8_eval_protocol.json` | `research/r9-new-test-readiness@f8e43b0` | inherited R8 protocol |
| `tools/r8_eval_manifest.py` | `research/r9-new-test-readiness@f8e43b0` via PR #11 tree | manifest build/verify/disjoint/freeze |
| `tools/r9_final_disjoint.py`, `tests/test_r9_final_disjoint.py` | PR #11 `93faece` | independent content-disjoint gate under verification |
| `report/R9_INDEPENDENT_CONTENT_DISJOINT_AUDIT.md` | PR #11 `93faece` | the audit note being verified |
| `tools/r9_contract_gate.py`, `tools/r9_env_snapshot.py`, `tests/test_r9_contract_gate.py` | `experiment/r9-freeze-64train@f7b8b39` | Gate A independent verification |
| `training/r9_trainer_contract.py`, `training/r9_train_pushT_svacc.py`, `tools/r9_trainer_trace.py`, `tests/test_r9_trainer_contract.py` | `feature/r9-vla-trainer-contract@1cd50f1` | Gate B 42D/R wiring + trace |

`data/split80.json` is the frozen 80/20 split (`2654e319…`). No HDF5, weights or secrets are copied.

Note: running the R9 KV regression in this tree rewrites its own output file
`evidence/r9_kv_cpu_20261020/test_r9_kv_same_prefix.json`; that file is restored before commit so the
R9 evidence stays byte-identical. The R10 copy of that run is `evidence/r10/test_r9_kv_same_prefix_r10tree.log`.
