# Gate A — failure log (原始失败留证)

All iterations kept: the checker was fixed, the GATE CRITERIA were not relaxed. Two harness
defects and one criterion refinement happened before the final 9/9 PASS.

| # | action | result | diagnosis | fix |
|---|---|---|---|---|
| 1 | `tools/r9_contract_gate.py` first run | `TypeError: Accessing a group is done with bytes or str, not tuple` | my independent reader used `h["state"]` while the dataset stores `state/<joint>_joint_states` | read the four documented joint groups in the 14D order |
| 2 | second run | `TypeError: encode_indices() got multiple values for argument mode` | production codec takes normalised values, not `(actions, q01, q99)` | call `AC.encode_indices(AC.normalize(...), mode=MODE_FIXED)` |
| 3 | third run | `codec_roundtrip_on_holdout` FAIL: max error 0.2648 vs one bin 1.72e-4 | criterion conflated BY-DESIGN tail clipping (q01/q99 cut 1%/side) with quantisation error | split the metric: in-range error must be <= 1 bin; out-of-range is reported as clipped fraction |
| 4 | final run | **9/9 PASS**, `tests/test_r9_contract_gate.py` 15/15 (7 negative controls) | - | - |

Note on iteration 3: the pass condition on in-range entries (<= one quantisation step) is the
same physical requirement as before; the change only stops counting the deliberate 1%/99% tail
clip as a codec defect. Out-of-range fractions are still reported (1.80% on the holdout).
