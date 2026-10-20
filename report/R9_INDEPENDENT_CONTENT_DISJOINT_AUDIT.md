# R9 independent content-disjoint audit (2026-10-20)

**Status: IMPLEMENTED + 9/9 LOCAL_SYNTHETIC_VERIFIED; remote real-HDF5 regression PENDING.**

Baseline: `research/r9-new-test-readiness@f8e43b0ca6ccaa8a5711e338ba12e0f3be69328f`.
Do not change `main`, original R9 evidence, `protocol/r9_eval_protocol.json` or the existing R8 checker.

## P0 finding: g3's original implementation is filename-only across datasets
`tools/r8_eval_manifest.py::check_disjoint(manifest, split_path)` checks:
- candidate filenames against used split filenames;
- duplicate content **within** candidate.
It does **not** compare candidate episode `content_sha256` against the 100 historical episode hashes.
A renamed copy of a previously used episode can pass g3 if the candidate has no internal duplicates.

## Narrow patch
- Add `tools/r9_final_disjoint.py` as an additional fail-closed **g3/g4 pre-freeze check**. Required arguments: candidate manifest, complete 100-episode `manifest_existing100.json`, `split80.json`, R9 protocol, and a new unused output JSON path.
- Reject candidate/old content-hash AND state/action-hash intersections, filename intersection, internal duplicates, incomplete or unanchored used reference, wrong task/frequency/length, and fewer than 5 DEV + 15 FINAL episodes. Verify the 100-episode reference dataset SHA against the protocol.
- The original R8 script is unchanged; this new check must be used before any future R9 FINAL freeze, and the execution protocol should explicitly include it on next revision.
- Add 9 synthetic positive/negative controls, notably **renamed old content with a new filename**.

Local synthetic regression (run by independent auditor, no HDF5/GPU/data access):
```
python -m py_compile tools/r9_final_disjoint.py tests/test_r9_final_disjoint.py
python tests/test_r9_final_disjoint.py
# 9 tests, OK
```

Future real-data g3/g4 command, AFTER authorized acquisition and manifest creation:
```bash
python tools/r9_final_disjoint.py \
  --candidate <NEW_MANIFEST.json> \
  --used-manifest evidence/r9_eval_prep_20261020/manifest_existing100.json \
  --split data/split80.json \
  --protocol protocol/r9_eval_protocol.json \
  --out <NEW_EMPTY_EVIDENCE_DIR>/r9_content_disjoint.json
```

**Residual gates**: This does not validate camera streams, HDF5 shape, file-size placeholder criterion or the freeze/commit order. Run official `verify`, explicit real-HDF5 inclusion checks, and Git SHA+hash freeze only after this P0 check. This audit is **not** an authorization to acquire data or evaluate FINAL.

## Additional findings outside this patch
- C: `diagnostics/r9_kv_same_prefix.py --gpu-diagnose` currently returns status 0 after printing a command; it does not actually run the GPU probe. Do not use wrapper exit 0 as GPU PASS. Invoke the guarded real `diagnostics/r8_smoke.py --kv-diagnose` in an authorized, bounded GPU window and require output logits and nonzero-on-mismatch.
- B: New training loop remains unverified on genuine 7B and needs action-dimension invariant and proper checkpoint metadata, plus bounded `--steps` authorization enforcement, before any optimizer step.
- C noise classifier: fixed `numeric_floor=0.05` is an arbitrary heuristic; the observed tiny-model fp32 noise is about `8e-8`, and the injected cache error `0.0439` is mislabelled `numeric_plausible`. It must not be interpreted causally; keep root cause OPEN.

No source or evidence on the four delivered R9 branches was modified.
