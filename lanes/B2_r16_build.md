# B2 - R16 build: the 42D OpenVLA-R42 deployment path (CPU-verified, GPU pending)

Scope: wire the finished 42D [S,V,Acc] Push-T policy (r11 repo,
code/xpolicylab/policy/OpenVLA_R42/) into the RoboDojo / XPolicyLab WebSocket policy-server
protocol, and make every step of that wiring executable and auditable on CPU while the GPU is
busy. No model was loaded, no checkpoint was modified, no process was touched, nothing was
committed.

## 1. Status of the audited blockers

| # | Blocker | Status | Evidence |
|---|---------|--------|----------|
| G1 | RoboDojo/XPolicyLab/policy/OpenVLA_R42 does not exist | **resolved on the server side** (no RoboDojo file needed) | code/xpolicylab/policy/OpenVLA_R42/{__init__,model,deploy,deploy.yml,setup_eval_policy_server.sh,r42_serve.py}; r42_serve.py builds the same PolicyServer from the r11 tree |
| G2 | r42_deploy_server imports r8_* / action_codec that only exist in r11 | **resolved by path bootstrap** | r42_deploy_server.py:24-30 inserts its own dir + ../OpenVLA_Pure (unchanged); model.py:34-38 and r42_env_io.py:41-44 do the same, so both trees resolve |
| G3 | no env-dict <-> flat-14D + 3-camera conversion | **done** | r42_env_io.py: obs_dict_to_inputs / cmd14_to_action_dict / action_dict_to_cmd14, 45 tests |
| G4 | R15 stage checkpoint has no r42_metadata.json | **done, byte-parity proven** | r42_metadata_bridge.py; reproduces all 7 existing r42_metadata.json files byte for byte |
| G5 | deploy forward != training forward | **done** | r42_deploy_server.py install_pure_autoreg() + _TorchBackend.next_token_logits(..., labels=ids.clone()); 5 tests |
| G6 | latency stamped server-side only | **done** | 4 phases asserted server-side (r42_deploy_server.py _check_latency), carried in the infer payload, client RTT + skew-free server span in r42_ws_timing.py |
| G9 | reset must reset all envs | **done** | model.py Model.reset() -> R42DeployServer.reset_all(); test_reset_clears_every_env_g9 |
| G12 | served codec mode must be printed/asserted | **done** | r42_deploy_server.py _assert_codec_mode + _announce; model.py re-asserts; test_metadata_with_a_non_fixed_codec_mode_is_refused |

## 2. What is now wired end to end (and proven on CPU)

    TASK_ENV.get_obs()
      -> obs_dict_to_inputs      vision{cam_head,cam_left_wrist,cam_right_wrist}["color"] uint8 HxWx3
                                 state{left_arm6,left_ee1,right_arm6,right_ee1} -> flat 14D
                                 'action' dropped, privileged fields refused, 25 Hz asserted
      -> Model.update_obs        (fails closed BEFORE any GPU work)
      -> R42DeployServer.act     42D [S,V,Acc] features -> 14 legal tokens -> S+V+R safety
      -> WsModelClient           msgpack WS round trip (latency_ms + stamp in the payload)
      -> Model.get_action        {"actions": {4 keys}, "latency_ms": {4 phases}, "stamp": {...}}
      -> cmd14_to_action_dict    exactly what eval_env.take_action() consumes

Wire proof: tests/test_openvla_r42_ws.py starts the real
client_server.ws.model_server.PolicyServer on an ephemeral port and drives it with the real
WsModelClient (the object the env uses) - 6 tests, including a malformed frame and a bad
observation that must not kill the server.

### Latency (measured, loopback, stub backend, 200 steps after 10 discarded warm-up)

    [T9] budget=40.0ms n=200 | server total p50=24.631 p95=45.986 p99=53.754 ms |
         client rtt p50=31.190 p95=57.596 p99=69.501 ms | net p50=4.814 ms

The server phases are the ones the shipped stack reports (server-side inference only); the
client RTT, the skew-free server span (stamp.server_span_ms, perf_counter) and net
(= rtt - span, serialisation + socket + client bookkeeping) are new. These are stub-backend
numbers on one host: they say nothing about 7B inference time.

## 3. Evidence

Test command and result:

    cd /home/T7/ojh/robot_actual/r11
    export ROBODOJO_ROOT=/home/T7/ojh/robot_actual/RoboDojo
    /home/dgut/pusht_runtime/venv_oft/bin/python -m pytest \
      tests/test_r42_env_io.py tests/test_r42_metadata_bridge.py \
      tests/test_r42_contract_parity.py tests/test_openvla_r42_server.py \
      tests/test_openvla_r42_ws.py tests/test_r42_panic.py \
      tests/test_r42_robodojo_adapter.py -q

    162 passed in 139.70s (0:02:19)

    file                                  tests  covers
    tests/test_r42_env_io.py                 45  T2 obs conversion, T3 action conversion (34 defs, parametrised)
    tests/test_r42_metadata_bridge.py        16  G4 bridge + byte parity + refusals
    tests/test_r42_contract_parity.py        17  T4 42D feature parity, T5 codec parity, T6 decode+safety parity
    tests/test_openvla_r42_server.py         19  T7 stub-backend server, G5, G12, G9
    tests/test_openvla_r42_ws.py              6  T8 full local WS end to end, T9 latency percentiles, wire panic
    tests/test_r42_panic.py                  23  T10 panic paths
    tests/test_r42_robodojo_adapter.py       36  pre-existing suite, still green (backward compatibility)

Metadata bridge byte parity (7/7 existing files, rebuilt from MANIFEST + contract + bounds):

    PARITY_OK r13_smoke_20261020T193912Z/checkpoints/step0   4365 bytes sha=60bb9ceebfd5288b
    PARITY_OK r14_overfit_20261020T194610Z/checkpoints/step0   4365 bytes sha=36b312a8acf5c054
    PARITY_OK r14_overfit_20261020T194610Z/checkpoints/step100 4367 bytes sha=efe033d516f6763a
    PARITY_OK r14_overfit_20261020T194610Z/checkpoints/step20  4366 bytes sha=a12024b52e2defe8
    PARITY_OK r14_overfit_20261020T194610Z/checkpoints/step40  4366 bytes sha=7ea526820b9834c6
    PARITY_OK r14_overfit_20261020T194610Z/checkpoints/step60  4366 bytes sha=ba401c81951ed106
    PARITY_OK r14_overfit_20261020T194610Z/checkpoints/step80  4366 bytes sha=db64d70d1e906deb

The R15 case (no metadata anywhere) on the real checkpoint
runs/r15_stage_300_20261021T030423Z/checkpoints/step300:

    train_steps=300        (manifest.global_step == trainer_state.global_step, both checked)
    lora_sha256=f0316a91...  == MANIFEST lora_adapter/adapter_model.safetensors
    projector_sha256=b71ec79f... == MANIFEST proprio_projector_42.pt
    base_model_sha256=bcb688a6... from trainer_state.base_model["config.json"] (file verified
                                  against the MANIFEST sha first)
    normalizer_sha256=13a80f91... == trainer_state.data_contract.normalizer_sha256
    bounds/contract/split sha256 all cross-checked against trainer_state.data_contract
    file_sha256=9f9354bcabdc5511e7e263d98f719321d34f4b1900408a90bbda5bb3c1b7d4ec

Server assembly without a GPU (proves G1 + G4 + G2 together, no weights touched):

    $ python code/xpolicylab/policy/OpenVLA_R42/r42_serve.py \
        --config_path code/xpolicylab/policy/OpenVLA_R42/deploy.yml \
        --overrides checkpoint_dir=$R11/runs/r15_stage_300_20261021T030423Z/checkpoints/step300 \
        --print-contract
    [OpenVLA_R42] template=XPolicyLab.model_template codec_mode=FIXED (asserted)
                  window=[31744, 31999] n_actions=14 camera_order=['cam_head','cam_left_wrist',
                  'cam_right_wrist'] control_hz=25 device=cpu
    "metadata_source": "r42_metadata_bridge(.../checkpoints/step300)"
    "template_source": "XPolicyLab.model_template"

Template resolution is honest both ways: with only the r11 repo on sys.path the model reports
template_source="builtin_standin" (a faithful 6-method stand-in, unit tested); with
RoboDojo/XPolicyLab on sys.path it subclasses the real XPolicyLab.model_template, as the
print above shows.

## 4. Files (additive unless marked)

    code/xpolicylab/policy/OpenVLA_R42/__init__.py                    1553   new
    code/xpolicylab/policy/OpenVLA_R42/model.py                      19959   new  (Model(ModelTemplate))
    code/xpolicylab/policy/OpenVLA_R42/r42_env_io.py                 18241   new  (G3 converters)
    code/xpolicylab/policy/OpenVLA_R42/r42_metadata_bridge.py        21801   new  (G4)
    code/xpolicylab/policy/OpenVLA_R42/r42_ws_timing.py              10392   new  (G6 client RTT)
    code/xpolicylab/policy/OpenVLA_R42/r42_serve.py                   4910   new  (split-mode launcher)
    code/xpolicylab/policy/OpenVLA_R42/deploy.py                      1185   new  (client loop)
    code/xpolicylab/policy/OpenVLA_R42/deploy.yml                     1106   new
    code/xpolicylab/policy/OpenVLA_R42/setup_eval_policy_server.sh    2365   new  (RoboDojo convention)
    code/xpolicylab/policy/OpenVLA_R42/r42_deploy_server.py          17946   MODIFIED (+117/-4: G5/G6/G12)
    tests/test_r42_env_io.py                                         12980   new
    tests/test_r42_metadata_bridge.py                                11726   new
    tests/test_r42_contract_parity.py                                14346   new
    tests/test_openvla_r42_server.py                                 14553   new
    tests/test_openvla_r42_ws.py                                     12540   new
    tests/test_r42_panic.py                                          10851   new
    lanes/B2_r16_build.md                                            12749   new (this file)

Untouched: training/, diagnostics/, tools/, mission/, scripts/, the RoboDojo tree, and
runs/ (the bridge only reads). r42_robodojo_adapter.py was already untracked from the earlier
audit lane and was not modified.

## 5. What still needs GPU / Simulator (NOT executed here)

1. R42DeployServer.load(): the real 7B + LoRA + 42D projector load. Everything downstream of it
   is tested with a stub backend; the load itself is GPU-only and was never run in this lane.
2. Forward parity (G5) on real weights: the patch and the labels= argument are tested
   structurally (a fake core proves labels is always passed and the patch lands on the core,
   not the PEFT wrapper); the token-for-token comparison of training vs deploy logits needs a
   GPU forward.
3. Closed-loop Push-T success rate: cannot exist without IsaacSim + the real weights.
4. Real 480x640 camera frames: the converter is tested with the exact audited shape/dtype, but
   on synthetic arrays.
5. Real-network client RTT: the T9 numbers are loopback with a stub backend.
6. The processor path (preprocess(), build_prompt()) - GPU/transformers only.

## 6. The two commands that would run one DEV episode end to end (NOT EXECUTED)

One-time client-side install (the only thing not done for you, because it writes outside the
allowed file set): the RoboDojo client resolves the episode loop at
XPolicyLab/policy/<name>/deploy.py (scripts/robodojo.sh:439-441, then
src/eval_client/eval_env.py:325-334 imports it), so link the r11 directory in:

    ln -s /home/T7/ojh/robot_actual/r11/code/xpolicylab/policy/OpenVLA_R42 \
          /home/T7/ojh/robot_actual/RoboDojo/XPolicyLab/policy/OpenVLA_R42

The CLIENT never imports model.py; it only needs deploy.py (+ deploy.yml for the protocol).

Server (GPU 0, real R15 stage checkpoint, ws on 19000):

    cd /home/T7/ojh/robot_actual/r11
    ROBODOJO_ROOT=/home/T7/ojh/robot_actual/RoboDojo \
    R42_PYTHON=/home/dgut/pusht_runtime/venv_oft/bin/python \
    bash code/xpolicylab/policy/OpenVLA_R42/setup_eval_policy_server.sh \
      "" push_T \
      /home/T7/ojh/robot_actual/r11/runs/r15_stage_300_20261021T030423Z/checkpoints/step300 \
      arx_x5 joint 0 0 "" 19000 0.0.0.0

Client (one DEV episode; EVAL_NUM is clamped to the task's eval_nums - 25 for push_T, see
src/eval_client/main.py:325-328 - so never ask for more than 25):

    cd /home/T7/ojh/robot_actual/RoboDojo
    bash scripts/robodojo.sh client --task push_T --policy-name OpenVLA_R42 \
      --policy-host 127.0.0.1 --policy-port 19000 --eval-num 1 --seed 0

Expected server-side first lines (G12 banner): "[R42DeployServer] codec_mode=FIXED (asserted)
action_token_window=[31744, 31999] n_actions=14 task=push_T ...".

## 7. Honest limitations

* Float32 re-encode drift: LEVELS sits exactly on a np.digitize bin edge, so decoding a bin and
  re-encoding it moves 615/3584 channels by exactly 1 bin (max drift 1). Deploy only decodes
  (bins -> residual -> command), and the decode direction is exact for all 256 bins; this is
  documented and bounded in tests/test_r42_contract_parity.py rather than hidden.
* The base-model sha256 is not in MANIFEST.json; the bridge reads it from trainer_state.json
  after verifying that file against the MANIFEST sha, or takes --base-model-sha256 /
  --base-model-config explicitly, and otherwise refuses.
* Model construction is lazy but strict: a checkpoint_dir without lora_adapter/ or
  proprio_projector_42.pt is refused at warmup instead of serving the pristine 7B.
* deploy.yml points at the step-300 stage checkpoint as a placeholder; change checkpoint_dir (or
  pass ckpt_name) to serve a different stage.
