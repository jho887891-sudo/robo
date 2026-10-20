# Gate B — failure log (原始失败留证)

Absolute-byte entrypoint is untouched; the trace harness needed three iterations before 11/11.

| # | action | result | diagnosis | fix |
|---|---|---|---|---|
| 1 | first trace run | B2/B3 FAIL, then `RuntimeError: Number of dimensions of repeat dims ...` in the mock head | (a) B2 required `q99 > q01` for every dim, but 2 episodes are too few so `left_gripper` degenerates; (b) B3 compared `assert_legal_token_ids(...) is None` although the helper can return True; (c) mock logits used `unsqueeze(0).repeat(N,1)` on a 3D tensor | require `q99 >= q01` and report degenerate dims; accept None or True; `z.repeat(N,1)` |
| 2 | second trace run | B10/B11 exit 0 but the print of details broke the script (`SyntaxError` in an unedited helper line) | my own patch introduced a broken string literal | rebuilt the `check()` printer, recompiled, re-ran |
| 3 | final run | **11/11 PASS**, `tests/test_r9_trainer_contract.py` 16/16 | - | - |

`B11` is not a failure: it records an OPEN RISK (prismatic constants resolve to LIBERO
`ACTION_DIM=7` in this environment). It is labelled HYPOTHESIS and does not gate Gate B.
