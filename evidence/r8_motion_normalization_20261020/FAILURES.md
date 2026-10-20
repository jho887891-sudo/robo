# Workstream A failure log (r8-motion-normalization)

## F1 - diagnostics crashed on the production normalizer signature

`r8_norm_diagnostics.dim_report` called `r8_contract.normalize_block` with per-dimension
bounds (shape (1,)) while the function requires 14D bounds -> ValueError. Fixed by computing
the zero-motion image analytically per dimension instead of slicing the production helper.

## F2 - sign_preserved had a boolean-index broadcasting bug

`zv[None, :][neg_v]` cannot index a (1,14) array with an (N,14) boolean mask -> IndexError on
the first real-data test (synthetic tests passed because they never hit that branch with N>1).
Fixed with `np.broadcast_to(zv, vn.shape)[neg_v]`. This is why the real-data regression test exists.

## F3 - two ad-hoc probe scripts failed (no result produced)

One had a Python syntax error, another used JSON keys from a different script schema. Both
crashed before printing; no number in the report comes from them.

## F4 - the audit script only covers train/val, not the 64/16/20 dev split

`diagnostics/r8_scale_audit.py` reports train and val only. A separate
`diagnostics/r8_norm_diagnostics.py` was added to cover train/dev/holdout with the same frame
definition (t >= 2). The audit script itself was not modified.

## F5 - remote patch/heredoc tooling failures

Several remote patches failed for quoting reasons. Fixed by uploading patch scripts as files and
py_compile-ing after every patch. No failed patch produced any measured result.
