
## R11–R16 first execution window (2026-10-20 19:20Z → 20:05Z, branch feature/r11-p0-closeout)

Executed, with real GPU evidence and a book-kept budget (0.285 GPU-hours of 32 h, formal optimizer
steps 0 of 3000):

* **P0 VERIFIED_CPU** — five source-level defects fixed red-first: the R42 hand-written codec was
  not bit-equal to `action_codec` (missing `+1e-8` in denormalisation) and the prompt lacked the
  legacy trailing `<space>` (29871) token, so the causal shift was one position early; the
  prismatic `ACTION_DIM=7` table (chosen from `sys.argv`) now fails closed unless it is ARX 14D,
  and the legacy `range(ACTION_DIM)` 7-token loop is pinned by a regression instead of being
  relied on; the rate limiter and the physical box can no longer disagree silently; the 21-field
  checkpoint metadata now carries the frozen normalizer and its canonical SHA; a real 42D deploy
  server replaces the CPU-only adapter. Suite: 250 passed, 3 skipped (baseline 208/3).
* **R13 VERIFIED_GPU_ZERO_UPDATE** — first real 7B 42D forward/backward in this project:
  loss 14.4278, LoRA grad norm 56.75, projector grad norm 16.17, `optimizer_steps = 0`,
  14 legal action tokens, common-prefix logit delta 0.0, checkpoint→reload→identical tokens.
  99.7 s, peak 22.16 GB. A first attempt crashed on a `torch.Size` bug after loading the 7B and
  its 70 s are also booked.
* **R14 OVERFIT_SIGNAL_PASS** — 100 steps on 32 fixed TRAIN frames from 32 distinct episodes:
  CE 14.814 → 1.908 → 1.305 → 0.541 → 0.380 → 0.125 at steps 0/20/40/60/80/100, training loss
  head mean 5.63 → tail mean 0.40, greedy 14-token decode 9 → 13 distinct bins. 857.8 s.
* **R11 collector BLOCKED_EXPERT** — read-only discovery found no executable Push-T expert in the
  Isaac Sim container (robodojo.sh drives policies under test; the 100 episodes were downloaded;
  the only "expert" file is a pybullet artefact). COLLECTOR bucket untouched, no synthetic data.
* **R15/R16 NOT_STARTED** — the staged trainer, gate machinery and ledger are implemented and both
  upstream gates are green, but the window ended before the 300/1000/3000 runs.
* **Draft PR not created** — the box has no GitHub API credential (no `gh`, no token); the branch
  is pushed and this is recorded rather than papered over with a `/pull/new/` link.
