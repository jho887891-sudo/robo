# R10 — failure / iteration log (原始失败留证)

No acceptance criterion was relaxed. Two test iterations exposed real gaps in **my** code, not in the
gates under test.

| # | what ran | result | diagnosis | fix |
|---|---|---|---|---|
| 1 | `tests/test_r10_kv_probe.py` | 15/16, `kv_diagnose_no_gradient_checkpointing_enable` FAIL | the assertion scanned the whole file, but the main smoke mode legitimately calls `gradient_checkpointing_enable()` for its backward pass | scope the assertion to the `kv_diagnose` block only |
| 2 | `tests/test_r10_freeze_guard.py` | FAIL `synthetic_set_passes_all_gates`, then aborted | (a) PR#11 `check()` enforces the protocol `final_episodes_min=15`, unreachable for a 6-episode synthetic set; (b) `M.verify` crashed when `M.build` had failed on the placeholder file | added an explicit, recorded `--allow-small-set-for-tests` override (test only, sets `dev_episodes`/`final_episodes_min`); guard now records `build` as its own check and skips verify with an explicit error |
| 3 | same test after fix | positive PASS but still aborted at negative 3 | placeholder file made `M.build` raise before the schema path | `build` wrapped, `cross_set_disjoint` and `verify` fail closed with recorded errors |
| 4 | final | **probe 16/16, guard 10/10** | - | - |

Resource findings that are *not* code failures: free VRAM 23.300 GiB < 24 GiB (GPU-KV -> BLOCKED_RESOURCE)
and no collection entry point in the checked-out RoboDojo (NEW-DATA -> BLOCKED_ENV). Both are recorded
with raw measurements in `gpu_resource_probe.json` and `report/R10_NEW_DATA_PROVENANCE.md`.
