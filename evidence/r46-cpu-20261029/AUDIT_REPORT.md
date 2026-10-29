# R46 CPU-only smoke preparation audit

Date: 2026-10-30 (review-fix gate update)
Repository: `jho887891-sudo/robo`
Branch: `codex/r46-smoke-supervisor-20261029`
Base: `ba563d0f8c638b089f588027dd0af41ff290e1f4`
Original implementation commit: `75b6a1f84d4af513472f20df5548117a5e908066`
Final reviewed CPU-gate code commit: `b7b32790ece97c560b1164eb96b472aa26c9dbde`

## Scope and safety

Prepared an independently supervised five-sample OpenVLA-7B smoke entrypoint. The frozen contract is a frozen standard OpenVLA-7B base, LoRA `r=32, alpha=16, q_proj/v_proj`, the 42D projector, three cameras in R44 order, `joint_delta_v1`, one warm-up sample and four measured samples. The child is limited to five disposable optimizer steps and exports no checkpoint or weights.

No model weights were loaded in this task. No CUDA context, GPU smoke, GPU benchmark, formal training, deployment, `main` change, or data/model/checkpoint write occurred. The sample freezer set `CUDA_VISIBLE_DEVICES=`; its output recorded `cuda_initialized=false`.

## VERIFIED

- **Final reviewed Linux CPU Gate:** clean detached worktree `/tmp/robo-r46-gate-b7b3279` at exact SHA `b7b32790ece97c560b1164eb96b472aa26c9dbde`; status clean. All **12/12 commands exited 0** and **93/93 tests passed with zero skips**. This supersedes the 88-test result below for current code. Raw command stdout/stderr/exit codes, environment, source SHA verification, runner logs and summary are in `linux-final-b7b3279/`.
- Environment: Python 3.10.12, NumPy 1.26.4, Torch 2.5.1+cu124, Transformers 4.46.3, PEFT 0.13.2, h5py 3.16.0, OpenCV 4.13.0. Runner set `CUDA_VISIBLE_DEVICES=`. No GPU Smoke, CUDA model load, throughput benchmark or formal training ran. This R46 CPU suite did not rehash the full 300-file HDF5 manifest.
- The final runner SHA256-checks 10 files, including `training/r44_train.py` and the argument-bridge regression test; all ten matched in the pinned Linux worktree. The gate log checksum verification also exited 0.

- **Historical baseline Gate only:** clean detached Linux worktree at `75b6a1f84d4af513472f20df5548117a5e908066`; 11 commands and 88 tests, valid for that SHA only and superseded for current code by the final reviewed Gate above.
- Linux CPU environment: Python 3.10.12, NumPy 1.26.4, Torch 2.5.1+cu124, Transformers 4.46.3, PEFT 0.13.2, h5py 3.16.0, OpenCV 4.13.0.
- Historical baseline breakdown: R46 contract (6), sample helper (3), supervisor (22); R45 runtime (9), toy resume (3), pipeline (12); R44 delta (15), contract (4), trainer core (14): 88 total. Current Gate adds four supervisor boundary tests and the direct R44 argument-bridge test, for 93 tests.
- The final POSIX supervisor tests exercised approval refusal, process start failure, normal exit, cooperative SIGINT, deadline SIGINT→SIGKILL, orphan cleanup limited to the owned session, unrelated-process isolation, GPU-admission refusal, model SHA refusal before GPU admission, and output collision. R45 runtime tests also confirmed the existing benchmark remains `BLOCKED_NOT_IMPLEMENTED` and the legacy smoke refuses without authorization.
- On the Linux CPU host, the production R45 reader and actual processor froze five distinct `push_T` TRAIN episodes. The resulting processor shape was `[1,18,224,224]`; each record contains three decoded RGB camera inputs, a finite 42D feature, legal 14 action tokens plus STOP, and the manifest HDF5 SHA. Sample-plan SHA256: `67458ab3c6b23518c8616ec229ff62ed9af51d3d6bc817d54b21cdd109937d4f`. The plan is `config/r46_smoke_samples.json`; raw freezer output and exit evidence are in `sample-freeze-final/`.
- Local Windows verification: syntax check passed; R46 tests reported 31 run, 20 passed and 11 Linux-only process-group cases skipped. Linux ran those cases instead.
- Historical `source_SHA256SUMS` covered eight files. The final reviewed runner's `source_SHA256SUMS` covered ten files and all matched on the server. The published evidence-root checksum manifest uses relative paths.
- Independently recomputed SHA256 over the exact Git blob bytes at commit `75b6a1f84d4af513472f20df5548117a5e908066`; all eight matched the Linux `source_SHA256SUMS`. Windows checkout bytes may use CRLF and are not the Linux execution-byte comparison.

## RED → GREEN fixes retained as evidence

- The first Linux process-supervisor run failed: the PDEATHSIG pre-exec hook had accidentally contained process-group signaling code, so child startup raised `SubprocessError`. The original failure is preserved in `cpu-gate-initial-fail/`. The signal code was moved into `_signal_verified_session`; the Linux supervisor suite then passed.
- Independent review found that the parent compared the approved model identity list but did not hash the actual model directory before GPU admission. A regression test failed before the fix (`model-identity-red/`); the parent now resolves exactly one forwarded `--base-model`, rehashes every identity file, verifies the three-shard index, and rejects mismatch before GPU probing. The regression test and all 22 Linux supervisor tests passed (`model-identity-green/` and final Gate).
- The report-schema test first failed against the prior implementation commit because `validate_smoke_report` did not exist; the captured import failure is in `report-schema-red/`. The CPU-only validator was added and wired into report writing; its positive fixture returns only `CPU_REPORT_SCHEMA_VALID`, not a GPU result.
- PR #53 review identified a separate argument-bridge defect: the child wrapper's `--max-wall-seconds` was incorrectly appended to `R44.parse_args()`, although the R44 parser does not define it. A direct bridge regression was first run RED (helper absent at import), then GREEN after extracting `_build_r44_argv`; the helper now keeps the supervisor's time budget out of the R44 CLI while preserving the outer approval/deadline checks. The test directly passes the generated argv through `R44.parse_args()` without CUDA. Fix commit: `a9f2618cd9ec2707a71dde43e07096c99ce9c3ea`. Its Windows CPU evidence is in `review-fix-a9f2618/`: 5/5 commands exit 0, 21 tests passed and 11 POSIX-only tests skipped. This does not replace or extend the earlier Linux 88-test Gate at SHA `75b6a1f...`; the updated Linux Gate remains pending.
- A second review pass fixed two CPU-reproducible safety defects in commit `beca55e633c27fea152d3db4aac0eb9ab2ba2c2a`: post-smoke GPU inventory allows only the current smoke PID; approval validation rejects a window with less remaining time than the requested hard limit and tests the exact-fit boundary. The final Linux Gate passed on `b7b32790ece97c560b1164eb96b472aa26c9dbde`; earlier RED logs remain preserved.
- The final clean-commit Gate was run after both fixes and report-schema tests; earlier failed logs remain for audit and are not presented as passing evidence.

## IMPLEMENTED

- `diagnostics/r46_smoke_supervisor.py`: external approval schema validation; source, manifest, sample and actual model-file identity verification; UUID/memory/compute-process admission; parent-owned evidence; a monotonic hard deadline capped at 1800 seconds; Linux process-session supervision with SIGINT/SIGKILL and PDEATHSIG.
- `diagnostics/r46_smoke_contract.py`, `diagnostics/r46_freeze_samples.py`, and `diagnostics/r46_openvla_smoke.py`: strict five-record TRAIN plan, deterministic CPU sample freezing, a CPU-tested report schema, and the real standard OpenVLA five-step smoke child. The child refuses direct launch before importing Torch unless the parent/session/device boundary is present.
- CPU regression tests and `.github/workflows/r45-runtime-cpu.yml` integration.
- Post-review fixes: `_build_r44_argv` in `diagnostics/r46_openvla_smoke.py` forwards only supported R44 options; `probe_gpu(..., allowed_compute_pids=...)` permits only explicitly allowed process IDs; approval windows must fit the entire requested runtime. CPU regression coverage is in `tests/test_r46_openvla_smoke_args.py` and `tests/test_r46_smoke_supervisor.py`.

## Source-path audit

- Existing R44 one-step entrypoint: `diagnostics/r44_gpu_smoke.py:96` (`main`). Its presence was retained; R46 adds the parent supervisor and the fixed 1+4 sample runner around the standard R44 model/data helpers.
- Formal standard OpenVLA model construction and LoRA checks: `training/r44_train.py:430` (`build_model`). CPU-only strict R44 input/data checks: `training/r44_train.py:252` (`load_and_validate_cpu_plan`). DEV path: `training/r44_train.py:705` (`evaluate_dev`).
- Reused production image reader: `training/r45_data.py:195` (`RuntimeReader`).
- Existing stable GPU benchmark remains fail-closed: `diagnostics/r45_gpu_bench.py:49` (`main`), which emits `BLOCKED_NOT_IMPLEMENTED` at line 85.
- R46 approval and process supervision: `diagnostics/r46_smoke_supervisor.py:53` (`validate_approval`), line 355 (`run_child`), and line 543 (`main`). Sample contract and report schema: `diagnostics/r46_smoke_contract.py:44` and line 158.

## BLOCKED / UNKNOWN

- The supplied approval document is a template (`template_only=true`, `approved=false`). R47 is `STOP_WAIT_GPU_APPROVAL`; this CPU work does not authorize running the smoke.
- The real five-step model loading/forward/backward/optimizer path has **not** been executed. GPU memory, CUDA-device identity, GPU latency/throughput, and model runtime compatibility remain UNKNOWN.
- The updated CPU Gate is VERIFIED on the final code SHA. Actual OpenVLA weight loading, CUDA forward/backward, GPU memory/throughput and smoke results remain UNKNOWN and require separate explicit GPU authorization. The earlier 88/88 result applies only to SHA `75b6a1f...`.
- R45's stable GPU benchmark remains `BLOCKED_NOT_IMPLEMENTED`. A five-sample preliminary smoke, if separately authorized and run, cannot substitute for stable throughput or a five-epoch estimate.
- R48 remains `NOT_RUN` because it depends on real R47 GPU evidence.

## Commands and evidence map

The parameterized runner is `run_cpu_gate.sh`. It requires `R46_CPU_WORKTREE` and `R46_CPU_PYTHON`; optional `R46_CPU_EVIDENCE_POINTER` prevents overwriting an older pointer. The final run used a clean worktree pinned to `b7b32790ece97c560b1164eb96b472aa26c9dbde`. Raw command logs and exit codes are in `linux-final-b7b3279/`; earlier baseline and RED evidence remain in their original directories.

`SHA256SUMS` at this directory's root covers all published evidence files except itself. The captured server checksum files retain their exact output, including the temporary evidence-directory prefixes. No HDF5 contents, model weights, credentials, or approval document were published.
