# R8 round failure and defect log

All failures are kept with their diagnosis. No acceptance threshold was lowered, no
failure evidence was deleted, and none of the audit code or its thresholds was edited.

## F1 — the audit runner refused my new branch

`scripts/run_r7.sh` allows only `audit/r7-state-dynamics-baseline` and
`review/r7-state-dynamics-audit-20261020`. My R8 worktree is on
`audit/r8-pretraining-readiness`, so the first `--all` launch stopped with
`[STOP] use an isolated R7 baseline/audit worktree`.

Fix: created a second, dedicated worktree on the review branch
(`/home/T7/ojh/robot_actual/r8/review`) and ran the audit pipeline there verbatim.
No guard, threshold or source file was modified.

## F2 — two self-inflicted process kills

`pkill -f r8_coverage_ablation` matched the `bash -c` wrapper that contained the same
string, killing my own remote session (exit -1, twice).

Fix: filter by process name instead of the full command line, e.g.
`ps -eo pid,comm,args | awk '$2=="python" && /r8_coverage_ablation/'`.

## F3 — first ablation budgets were too small to be informative

The first run used 1,200 equal-update steps and 30 equal-epochs, far below the
convergence scale of the full-data model (~7,600-15,000 steps), so every configuration
was measured mid-training.

Fix: relaunched with 8,000 equal updates and 200 equal epochs. The superseded numbers
are not reported as results.

## F4 — the 200-frame subset covered only 50 of 64 episodes

The first ordering cycled episodes inside each stratum, so a 200-frame prefix reached
only 50 episodes (4 frames each), confounding "less data" with "fewer episodes".

Fix: coverage-first ordering (one frame per episode per outer pass, per-episode rotation
over motion strata). Now 200 / 2,000 / 19,127 frames cover 64 / 64 / 64 episodes with at
least 3 / 31 / 153 frames each, and the prefixes stay strictly nested
(`r8_verify_ablation.py` re-derives the sets and asserts A subset B subset C).

## F5 — the audit verifier's pilot200 claim fails in the audit's own single-directory layout

`scripts/run_r7.sh --all` writes the main grid and the `--restrict-t2` acceleration grid
into the SAME `R7_RUNS` directory. Both save `pred_pilot200_R_<seed>.npz` (the filename
encodes neither the frame set nor the epoch budget), so the second run overwrote the
first run's sidecars. The verifier compared 386-epoch / t>=2 sidecars
(0.023098 / 0.027960 / 0.022808) with the 377-epoch / t>=1 JSON
(0.023284 / 0.028404 / 0.022758) and reported 17/18.

Discrepancy per seed: 1.9e-4 / 4.4e-4 / 5.0e-5 - an artifact collision, not a numeric
error (the reconstruction cmd = S + V + R_hat is exact, max deviation 0.0).

Fix (without touching audit code): re-ran the main grid alone into
`/home/dgut/pusht_runtime/r7_supp/pilot_consistent`, composed a verification directory
holding the main run's pilot/SV sidecars together with the acceleration run's SVAcc
sidecars, and re-ran `r7_verify_claims.py` -> 18/18 PASS. Both the failing 17/18 artifact
and the passing 18/18 artifact are kept.

## F6 — probe scripts with wrong JSON key names

Two analysis scripts crashed on `KeyError` because the aggregate keys of
`r7_state_mlp.py` are `correction_R2` / `cosine_mean` / `sign_at_1e-3` while the new
ablation script uses `R_corr` / `R_sign`. Fixed by reading each JSON's own schema; no
result was computed from a failed script.

## F7 — one R8 contract test asserted the wrong slice

`test_env_isolation` compared `other[14:]` (28 values) with a 14-value expectation. The
code was correct; the test slice was wrong. Fixed by asserting the two 14-value blocks
separately.

## F8 — remote heredoc / quoting failures while patching

Three remote patches failed for tooling reasons: an IndentationError from a scripted
`str.replace`, a template-interpolation error on a shell variable inside a report body,
and a broken multi-line heredoc. None produced a result.

Fix: upload patch scripts as files and run them, keep reports free of raw shell
interpolation, and always `py_compile` after patching.
