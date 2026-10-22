"""Client-side episode loop for OpenVLA_R42 (runs in the RoboDojo environment process).

The client never imports the model: it talks to the policy server over the WebSocket protocol
through model_client.  This loop has the same shape as XPolicyLab/policy/OpenVLA_Pure/deploy.py
but steps ONE action block at a time (num_open_loop_steps: 1) and passes the server's 4-key
absolute-joint dict straight to the simulator:

    get_action() -> {"left_arm_joint_state": (6,), "left_ee_joint_state": (1,),
                     "right_arm_joint_state": (6,), "right_ee_joint_state": (1,)}

which is exactly what eval_env.take_action() / validate_action_dict() expect (eval_env.py:603).
"""


def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        action = model_client.call(func_name="get_action")
        TASK_ENV.take_action(action)


def eval_one_episode_batch(TASK_ENV, model_client):  # pragma: no cover - single env only
    raise NotImplementedError("OpenVLA_R42 serves a single environment per policy server")
