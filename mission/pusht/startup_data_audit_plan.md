# Push-T startup data audit: plan and interpretation contract

Status: IMPLEMENTED (CPU audit tool + unit tests); real HDF5 execution PENDING.
Branch: \`research/pusht-startup-data-audit\`, based on \`research/r25-seen-exposure@a505d98a2a1ce67477f24677fcfaba829c3357f2\`.
Scope: existing Push-T data and the real M2 step-2000 checkpoint only. No training, GPU forward pass, simulation, data collection, or checkpoint mutation.

## 1. Why this audit is necessary

The R27 read-only server task completed and posted its evidence in GitHub Issue #38. It verified the real checkpoint at:

\`/home/T7/ojh/robot_actual/runs/m2_two_2000_20261021T123314Z/checkpoints/step2000\`

The independently verified facts are:

- global step and sampler position: 2,000;
- sampler pool: 61,171 ordered task/frame slots;
- 2,000 distinct consumed frame identities;
- zero selected DEV frames in the consumed prefix;
- the R24 frozen TRAIN diagnostic set contains 136 selected frames, of which only 3 were actually consumed: 2 Push-T and 1 Stack Bowls.

Therefore, R24's earlier comparison of its selected TRAIN frames against DEV did **not** establish that the model failed to fit frames that had actually received gradients. The selected TRAIN diagnostic set was mostly TRAIN-UNSEEN. That invalidates the use of this specific comparison as decisive evidence against the training-coverage hypothesis. It does not prove that more training will solve the behavior.

Reference: https://github.com/jho887891-sudo/robo/issues/38

## 2. Confirmed training data path

The existing M2 trainer uses the sampled episode/frame to read:

- three JPEG RGB streams from \`vision/<camera>/colors[t]\`;
- 14D state and action arrays;
- the 42D proprioceptive feature built from state history;
- the residual target encoded into legal action tokens.

The code path in \`training/m2_train_shared.py\` calls \`_episode_images(..., t, ...)\`, then passes \`pixel_values=pv\`, \`proprio=pr\`, and action labels to the VLA forward call. The loss is backpropagated and \`opt.step()\` is executed. Images are part of the actual training path; the concern is **which image frames were exposed, whether the targets are aligned, and whether the sampled examples carry a learnable startup signal**.

Relevant source:
- \`training/m2_train_shared.py\`: \`_episode_images\`, \`_read_state_action\`, training loop and sampler.
- \`training/m2_multitask_data.py\`: episode schema, TRAIN/DEV isolation, task frame enumeration and residual definition.
- \`training/r42_train_contract.py\`: \`A[t] == S[t+1]\` target contract.

## 3. Target contract and scope limits

For an episode with state sequence \(S_t\) and absolute next-step target \(A_t\):

\[
A_t \approx S_{t+1}, \qquad
V_t = S_t-S_{t-1}, \qquad
R_t=(A_t-S_t)-V_t.
\]

When the alignment contract holds, \(R_t\) is the second difference of the state sequence. The trainer's valid history-safe target indices are \(t=2,\ldots,T-2\).

The HDF5 contract exposes state, action, and RGB arrays, but no established privileged object pose/contact signal in the known M2 schema. The audit may describe robot kinematics and RGB pixel variation, but must **not** label an episode phase as “approach” or “contact” merely from joint velocity or image difference. True task-stage labels require an independently verified privileged state source.

The code checks common frame counts and the A/S next-step relation. No per-stream wall-clock timestamps are part of the known contract, so it cannot prove physical camera/state synchronization.

## 4. Implemented audit

Tool: \`tools/pusht_startup_data_audit.py\`
Tests: \`tests/test_pusht_startup_data_audit.py\`

The tool is CPU-only and read-only. It:

1. validates the frozen Push-T split and the HDF5 episode inventory;
2. verifies the M2 checkpoint manifest and reconstructs the exact consumed sampler prefix from \`sampler.json\`, \`trainer_state\`, the frozen split, and the task-mix contract;
3. assigns valid frames to TRAIN-SEEN, TRAIN-UNSEEN, or DEV groups, rather than treating every frame in a TRAIN episode as seen;
4. audits A[t] versus S[t+1], residual distribution, arm-joint transition magnitudes, gripper transitions, and the first 25 transitions (one second at 25 Hz);
5. uses explicit sensitivity thresholds for near-stillness, labelled as exploratory rather than canonical physical labels;
6. checks JPEG validity and decodability for the first startup window of each episode and the exact TRAIN-SEEN frame indices, while summarizing adjacent-pixel differences only as a visual-change proxy;
7. emits per-episode state/action content hashes, split provenance hashes, checkpoint manifest hash, sampler order hash, and group-level summaries;
8. writes only to a new output file. It refuses to overwrite existing evidence.

The script does not load model weights into Torch, initialize CUDA, run inference, change HDF5/checkpoints/WAL, or take optimizer steps.

## 5. Fixed remote command

Run on the original data server, from a clean worktree at the exact PR HEAD. Use the actual HDF5 root proven by the M2 task loader:

\`\`\`bash
export CKPT="/home/T7/ojh/robot_actual/runs/m2_two_2000_20261021T123314Z/checkpoints/step2000"
export HDF5_ROOT="/home/T7/ojh/robot_actual/data/data/hdf5"
export OUTDIR="$(mktemp -d /tmp/pusht-startup-audit-XXXXXXXX)"
export OUT="$OUTDIR/report.json"

python tools/pusht_startup_data_audit.py \\
  --checkpoint "$CKPT" \\
  --hdf5-root "$HDF5_ROOT" \\
  --splits data/m2_splits.json \\
  --normalizer data/m2_task_normalizer_2task.json \\
  --start-window 25 \\
  --out "$OUT" \\
  > "\${OUT%.json}.stdout.log" \\
  2> "\${OUT%.json}.stderr.log"
status=$?
printf 'exit_code=%s\noutput=%s\n' "$status" "$OUT"
sha256sum "$OUT" "\${OUT%.json}.stdout.log" "\${OUT%.json}.stderr.log"
\`\`\`

The temporary output directory is created exclusively before the command; the report file itself must not exist, and the tool refuses to overwrite. If the remote checkout has uncommitted changes, do not reset or overwrite them: create a clean temporary worktree at the PR's exact HEAD.

## 6. Read the results in the correct order

### Gate A: source and target integrity

- TRAIN/DEV episode sets are disjoint and exactly cover the Push-T inventory.
- Every inspected state/action array is finite and 14D.
- Record exact equality count and rows exceeding \(10^{-6}\) for \(A_t\) versus \(S_{t+1}\).
- All sampled JPEGs decode, and camera frame counts follow the HDF5 contract.
- The report explicitly lists whether timestamp-like or pose/goal-like datasets were found.

Any integrity failure means the behavior analysis is blocked until the source issue is understood. Do not silently drop episodes.

### Gate B: actual exposure

- The reconstructed checkpoint sampler position must equal global step 2,000.
- Sampler order hash, task mix, and checkpoint manifest must validate.
- Report actual Push-T TRAIN-SEEN frame count, consumed duplicate count, and per-episode exposure.
- DEV exposure must be zero.
- The frame groups in the report must remain TRAIN-SEEN / TRAIN-UNSEEN / DEV.

### Gate C: startup signal characterization

For every episode, the report summarizes the first 25 state transitions and valid residual targets. It includes a sensitivity curve over arm-joint RMS thresholds \(10^{-5},10^{-4},10^{-3},10^{-2}\) radians per sample. These thresholds are not a ground-truth “robot is stopped” label; they show whether the conclusion is robust to threshold choice.

Compare:
- TRAIN-SEEN vs TRAIN-UNSEEN startup velocity and residual distributions;
- TRAIN-SEEN vs TRAIN-UNSEEN residual magnitude and signed per-dimension means;
- DEV distribution as a held-out descriptive comparison;
- valid image decode counts and adjacent pixel-change summaries;
- episode-level variation, not only frame-weighted means.

If the selected training frames barely cover startup, this is evidence for a coverage/sampling issue. If startup transitions exist but residual labels are near zero or inconsistent across episodes, investigate target semantics, demonstration behavior, and temporal alignment. If labels have signal but the model's predictions do not track it, move to a controlled visual-input/model prediction diagnostic.

No one result alone proves the cause.

## 7. Next experiments after this audit

1. **First:** inspect this report and the exact source hashes; no training yet.
2. **Second:** build a frozen, episode-disjoint prediction comparison using actual TRAIN-SEEN frames, TRAIN-UNSEEN frames, and DEV frames. Compare a no-residual baseline, the current model, and a simple TRAIN-only predictor under the same codec and SAFE decoder. Do not use DEV to fit any predictor or normalizer.
3. **Third:** run a visual ablation only if it can be done with the real model and a frozen frame set: identical labels and state features, comparing real RGB versus a clearly specified image control. A state-only linear probe cannot establish that RGB contains no action signal.
4. **Fourth:** decide whether to change the sampling distribution, collect additional startup demonstrations, change the action target, or train longer. The decision must be tied to the measured signal, not loss alone.

## 8. Decision boundaries

- **VERIFIED:** source contract, sampler reconstruction, target alignment and sampled image integrity.
- **HYPOTHESIS:** any claim that startup data is insufficient, the residual target is intrinsically unlearnable, or more steps will solve initiation.
- **BLOCKED:** missing HDF5, checkpoint/sampler mismatch, camera decode failures, split contamination, or missing required schema.
- **Not measured by this audit:** task success rate, physical approach/contact, model generalization, or closed-loop success.

The output is a data diagnosis, not a model verdict. Preserve every report and log; never overwrite R24/R27 evidence.
