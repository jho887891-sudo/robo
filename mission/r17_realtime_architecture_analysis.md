# R17 realtime architecture analysis: from 1582 ms to 40 ms (workline E)

No training was run for this document. It is grounded in the latency decomposition measured on the real
7B in this project, and it is written to be falsifiable: every option states what it changes, what it
would cost, and the cheapest experiment that would confirm or kill it.

## 0. The measured budget (real 7B, step2000, 64 timed actions)

| component | measured | note |
|---|---|---|
| full recompute, 14 tokens (one control step) | **5060 ms** | 14 x full multimodal forward |
| one multimodal forward (prefill, ~805 tokens) | **~361 ms** | 5060 / 14 |
| KV-cached control step (1 prefill + 13 increments) | **1145 ms** | implies **~60 ms per incremental token** |
| live closed-loop step, cached (incl. sim + websocket) | **1582 ms** | p50 over 600 steps |
| vision tower share of the forward | **~0 %** | sharing it was bit-exact and gave **1.0002x** |
| 25 Hz budget | **40 ms** | |

The decomposition is unambiguous: **the cost is the language-model forward over the multimodal sequence,
and it is paid once per action token.** Two independent handles follow: reduce the number of tokens, or
make each token cheaper. Everything below is one of those two.

## 1. Option matrix

| option | mechanism change | compute reduction | changes training distribution? | effect on 14D accuracy | checkpoint compatible? | cheapest test |
|---|---|---|---|---|---|---|
| **A. low-freq VLA + high-freq classical controller** | VLA emits a target at 2-5 Hz; a joint-space interpolator / PD tracks it at 25 Hz+ | none on the model; adds ~0 cost | no (the policy still sees one observation per VLA step) | latency becomes the controller's; tracking error adds to the action error | **yes, unchanged** | implement the interpolator in the server and re-run one DEV episode; compare motion curve + object displacement |
| **B. action chunking** (predict K actions per call) | one decode produces K x 14D instead of 1 x 14D | per action = (prefill + K x token)/K; at K=25: (361+24x60)/25 = **72 ms/action** | yes: the model must be supervised on K-step windows (the ARX chunk size is already 25 in the runtime contract) | unchanged if the head is trained on the same codec; compounding error over the chunk is new | **no** (needs retraining with chunked targets) | measure how fast accuracy degrades when the current 1-step model is queried K times open-loop |
| **C. non-autoregressive action head** | all 14 (or 14 x K) tokens from ONE forward, no token loop | (361 ms for 14 dims) -> **25 Hz achievable at K>=25** with 2.7x margin | yes | risk: the per-dim correlation the AR loop exploits is lost; must be trained | **no** | train a small head on frozen features and compare 14D L1 to the AR decode on the same DEV frames |
| **D. distillation to a 3B student** | student reproduces teacher actions | ~2.3x fewer FLOPs -> prefill ~157 ms | yes (student learns the teacher, not the demos) | bounded by the teacher, which currently **loses to constant velocity** - see the audit | no | none worth doing until the teacher beats CV, otherwise the student inherits a non-skill |
| **E. feature / prompt reuse** | reuse vision+prompt KV across a short window | up to ~30 % of the prefill (361 -> ~250 ms) | no | none (identical maths) | yes | already measured as insufficient: 1145 -> ~1030 ms |
| **F. shorter visual context** | fewer vision tokens (e.g. 1 camera, or pooled patches) | roughly proportional to vision tokens; but vision is not the bottleneck, so this helps only via attention length | yes (the model was trained with 3 cameras / 769 tokens) | likely degrades - the 3 cameras are the only depth cue | yes for the 1-camera ablation | ablate cameras offline on DEV and measure 14D L1 before touching the serving path |

## 2. What this rules out

* **"Fix the cache and you get 25 Hz"** - the cache is already correct and gives 4.42x; the remaining
  1145 ms is irreducible for a 14-token autoregressive decode of a 7B over ~800 tokens.
* **"Optimise the vision tower"** - measured: bit-exact sharing yields 1.0002x.
* **"Distill first"** - the teacher currently commands a residual with 0.47x the demonstration magnitude
  and R² ≈ 0. Distilling a non-skill produces a faster non-skill.

## 3. Recommendation, in order

1. **Option A now** (no training, no architecture decision): serve the existing checkpoint at 2-5 Hz and
   track its 14D target with a 25 Hz interpolator. This is the only route that reaches a 40 ms *control*
   period with today's weights, and it is honest about what it is: the VLA decides, the controller
   executes. It is exactly what the current synchronous simulation already tolerates (the sim advances
   one step per policy call), so the DEV loop can be re-run immediately to measure whether smoother
   tracking changes the object trajectory.
2. **Then C+B together, if and only if the physical gate is fixed**: the audit says the binding
   constraint is *capability*, not speed. A non-autoregressive chunked head is the natural architecture
   (it is also what makes 25 Hz possible), but it must be trained, and training the current recipe for
   more steps has already been shown not to move physical error.
3. **D (distillation) only after the teacher beats constant velocity.**

## 4. What needs a research-lead decision

Options B, C and D change the training objective/architecture and therefore the authorised training scope.
They are **not** started here. Option A is inference-side only and continues autonomously.

## 5. The uncomfortable conclusion, stated plainly

At 1582 ms per step the current stack is 40x away from 25 Hz, but **speed is not what is blocking task
success**. The 600-step Push-T run had 600 steps of a valid 600-step rollout: time was sufficient. It
failed because the commanded action is a weak, misdirected residual (see
`mission/r17_pushT_failure_mechanism.md`). Making this loop 40x faster would produce the same failure 40x
sooner.
