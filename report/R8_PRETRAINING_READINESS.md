# R8 pretraining readiness — is 42-dim history + velocity residual worth it?

Scope: this round is CPU-only. No OpenVLA-7B training was started, no GPU job was
touched, and the GPUs on the host stayed with the sessions already using them.

## 1. What R8 proposes, and what has to be true for it to work

    V_t    = S_t - S_{t-1}
    Acc_t  = S_t - 2 S_{t-1} + S_{t-2}
    X_t    = [S_t, V_t, Acc_t]                 42 dims, past-only
    R_t    = (A_t - S_t) - V_t                 supervision
    A_hat  = S_t + V_t + R_hat                 recovery

Eight requirements follow. Each is checked below with either a test or a measured
number.

| # | Requirement | Status |
|---|---|---|
| 1 | 42-dim input built from past observations only | verified (test) |
| 2 | history time alignment (V, Acc) exact | verified (test) |
| 3 | no future information leaks | verified (test) |
| 4 | first and second frame boundary rule, identical in training and deploy | verified (test) |
| 5 | velocity residual encode/decode through the shared 256-bin codec | verified (test + measured error budget) |
| 6 | physical action recovery A = S + V + R | verified (test) |
| 7 | streaming deploy equals batched training construction | verified (test) |
| 8 | legacy 14D / 28D contract still intact | verified (test) |

## 2. New artifacts (this branch, CPU only)

| File | Purpose |
|---|---|
| diagnostics/r8_velocity_residual_contract.py | the contract: proprio42, residual, recovery, train-only bounds, codec round-trip, observed-history buffer |
| tests/test_r8_contract_agent.py | 15 CPU contract tests, including a real-HDF5 minibatch test |
| a small pytest shim + run_contract_tests.py | the shared venv has no pytest; the shim implements only raises/approx/mark.skipif |

Run: python tools/run_contract_tests.py tests/test_r8_contract_agent.py
Result: 15 passed, 0 failed, 0 skipped (log: evidence/.../r8/contract_tests.log).
Module sha256 4aa7b4c964b6aa3ea4b375381990da47f8fdfa91048abaaa01b59694d997215b,
test sha256 006785700ec2bba5095630644ca6dbfc02d565b327412446058acb7db2716a3e.

## 3. Three findings that change the R8 implementation plan

### 3.1 The shared 256-bin codec floor-quantises, so the error budget is one bin

action_codec.encode_indices() does digitize(v, LEVELS) and then subtracts one: it
truncates towards -inf instead of rounding to the nearest level. The per-channel
decode error is therefore one full bin, (q99-q01+1e-8)/255, not half a bin.
Measured on the push-T residual: the worst channel error equals exactly
(q99-q01)/255 — ratio 1.000 against the naive half-step formula.

Consequence for R8: the target normalisation window must be chosen with a
full-bin error budget. The contract module now returns that bound explicitly
(quantization_error_bound) instead of the optimistic half-step.

### 3.2 Per-channel q01/q99 windows can be degenerate or far too narrow

On TRAIN residuals: channel 6 has a window of 3.7e-16 (a point) and channel 13
has 5.6e-6, while their residuals reach 5.2e-6 and 1.0e-1 respectively. A naive
q01/q99 fit therefore either divides by ~0 or clips almost the whole channel.
Both are handled explicitly now:

- windows narrower than 1e-6 are widened to a declared half-width (fit_bounds_on_train);
- window_coverage() reports, per channel, the covered fraction and the
  max/window ratio, so a channel that is effectively truncated cannot pass
  unnoticed. Measured TRAIN coverage is at least 0.90 per channel.

Any R8 run must print this table; the earlier 14D absolute-action contract did not
need it because those channels were not residual-scale.

### 3.3 The existing proprio history module cannot express 42 dims

code/xpolicylab/policy/OpenVLA_Pure/proprio_history.py supports three modes only:
none (14D), duplicate (28D) and true (28D = S_t with S_{t-1}). A 42-dim
ProprioProjector therefore needs:

1. a new mode (for example true3) with PROPRIO_DIM 42 and slot layout
   [S_t | S_{t-1} | S_{t-2}], keeping the rule that every slot uses the same
   train-derived pmin/pmax;
2. a two-deep history buffer instead of the current one-deep one, with reset()
   dropping both previous states;
3. checkpoint metadata recording proprio_dim = 42, the layout tag and the target
   mode R, so a 28D or 14D checkpoint cannot be loaded into a 42D projector by
   accident. verify_projector_width() already exists and would need the new mode.

None of this is done in the audit branch; this report states the required shape of
the change and proves the arithmetic with tests.

## 4. Why the target is worth trying (Stage 1 evidence)

The 42-dim arm is the best CPU result in the R7 reproduction, on the same holdout
episodes and the same budget:

| arm | L1 | R2 of the correction | direction sign | episodes beating CV |
|---|---|---|---|---|
| constant velocity (baseline) | 0.001828 | n/a (R = 0) | - | - |
| SV (28D) / R | 0.001727 | 0.311 | 0.957 | 14 / 20 |
| SVAcc (42D) / R | 0.001204 | 0.522 | 0.973 | 20 / 20 |

The acceleration term adds real information about the velocity correction: 30 %
lower L1 than SV, R2 from 0.311 to 0.522, and every holdout episode beats the
baseline in all three seeds.

Caveat that must travel with that number: the 20 holdout episodes have been used
repeatedly across R2–R7 to choose recipes, so anything reported on them is an
upper bound of what a fresh episode set would show. A final evaluation must use
episodes that no recipe has ever touched.

## 5. Go / no-go

| Gate | Status |
|---|---|
| contract tests green on real data | yes (15/15) |
| 42D arm beats constant velocity on the holdout | yes (0.001204 vs 0.001795, 20/20 episodes) |
| correction R2 above 0.3 | yes (0.522) |
| direction agreement above 0.75 | yes (0.973) |
| seeds stable | yes (three seeds, 0.001156–0.001233) |
| key joints not degraded | partial: DEV kept constant velocity for gripper dims 6 and 13; per-joint table is in state_mlp_acc.json |
| no leakage in the CPU contract | yes (past-only test, 64/16/20 disjoint) |
| model predicts a correction, not zero | yes: mean|R_hat| / mean|R_gt| = 0.88–0.92 with R2 0.52 |

Conclusion: worth a small GPU experiment, but it must not start from the audited
code as-is, because the 42-dim path does not exist there yet. The minimum GPU work
is (a) the 42D projector plus two-deep history buffer, (b) the R target with the
corrected error budget, (c) a checkpoint metadata gate that refuses mismatched
proprio widths, and (d) a fresh, never-used episode set for the final number.

## 5b. Data volume is not negotiable (Stage 2 evidence)

The volume ablation in the companion report changes the GPU plan in one important
way: no small pilot will predict the full-data behaviour.

| frames | best stable holdout L1 | R2 | episodes beating CV |
|---|---|---|---|
| 200 | 0.011590 (uniform, 8000 steps); 0.023284 (audited fixed 377 epochs) | negative | 0/20 |
| 2000 | 0.003009 (uniform) / 0.025719 (stratified) | negative | 0/20 |
| all 19,127 | 0.001204 (SVAcc) / 0.001866 (SV) | +0.52 / +0.28 | 20/20 / 7.3 of 20 |

Two traps to avoid when the GPU stage starts:

1. **Do not run a 200-frame or 2,000-frame OpenVLA pilot to "check the idea".**
   In the CPU analogue neither size ever beats constant velocity, and with a long
   budget both diverge; a GPU pilot at those sizes would burn time and produce a
   false negative.
2. **Do not stratify the sample by |V|.** It made the 2000-frame arm 8.5x worse
   than uniform sampling at the same size (0.0257 vs 0.0030), and it diverged at
   200 frames where uniform stayed finite. If stratification is wanted, stratify
   on the residual distribution and keep the marginal weights.

The GPU stage should therefore train on all legal training frames, with a
standardisation guard (clip or floor the per-dimension spread) because the CPU
runs showed that a tiny-spread channel can blow the network up independently of
the learning rate.

## 6. What is still unknown

- Whether a 7B model with a 42D projector keeps the CPU MLP advantage; the CPU
  result shows the information is present, not that the large model will use it.
- KV-cache autoregressive inference was exercised only as a token round-trip
  contract; the real 7B decode path was not run this round.
- Whether the DEV preference for constant velocity on the gripper channels (6, 13)
  is a data artifact or a real do-not-learn signal needs its own experiment on
  fresh episodes.
- Physical deployment timing (observation latency, state alignment on the robot)
  is untouched.
