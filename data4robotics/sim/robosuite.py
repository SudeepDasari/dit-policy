# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from collections import deque

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
import tqdm

from data4robotics.sim import SimTask, SimTaskReplayBuffer

os.environ["MUJOCO_GL"] = "egl"
OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
]
_MAX_STEPS = 200
_N_ROLLOUTS = 50
EXP_WEIGHT = 0.0


_AC_LOC = None
_AC_SCALE = None


def _normalize_actions(actions):
    loc = _AC_LOC[None].astype(actions.dtype)
    scale = _AC_SCALE[None].astype(actions.dtype)
    return (actions - loc) / scale


def _denormalize_action(actions):
    return actions * _AC_SCALE + _AC_LOC


def _render(env, height=256, width=256):
    img = env.render(
        mode="rgb_array", height=height, width=width, camera_name="agentview"
    )
    return img.astype(np.uint8)


def _obs_dict_to_vec(obs_dict):
    return np.concatenate([obs_dict[k] for k in OBS_KEYS]).astype(np.float32)


def _get_task_path(task):
    # little hack now for transport
    global OBS_KEYS, _MAX_STEPS
    if task == "transport" and "robot1_eef_pos" not in OBS_KEYS:
        OBS_KEYS += [
            "robot1_eef_pos",
            "robot1_eef_quat",
            "robot1_gripper_qpos",
            "robot1_gripper_qvel",
        ]

    if task in ("transport", "tool_hang"):
        _MAX_STEPS = 725

    task_path = os.path.expanduser(f"~/robomimic/datasets/{task}/ph/low_dim.hdf5")
    assert os.path.exists(task_path), "Missing task data!"
    return task_path


def _make_env(task):
    task_path = _get_task_path(task)
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=task_path)

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=True)
    env.reset()
    return env


class RoboSuiteBuffer(SimTaskReplayBuffer):
    def load_buffer(self, task):
        global _AC_LOC, _AC_SCALE

        render_env = _make_env(task)
        task_path = _get_task_path(task)
        buffer = []
        all_actions = []
        with h5py.File(task_path, "r") as f:
            print("Loading demonstration data!")

            for demo in tqdm.tqdm(list(f["data"].keys())):
                sim_states = f[f"data/{demo}/states"][()]
                demo_acs = f[f"data/{demo}/actions"][()]

                images, obs, actions = [], [], []
                for s, a in zip(sim_states, demo_acs):
                    # reset to demo state and generate robot obs/images
                    obs_dict = render_env.reset_to({"states": s})
                    obs.append(_obs_dict_to_vec(obs_dict))
                    images.append(_render(render_env))
                    actions.append(a.astype(np.float32))
                    all_actions.append(a.astype(np.float32))

                buffer.append(
                    dict(
                        images=np.array(images),
                        observations=np.array(obs),
                        actions=np.array(actions),
                    )
                )

        all_actions = np.array(all_actions)
        max_ac = np.max(all_actions, axis=0)
        min_ac = np.min(all_actions, axis=0)
        _AC_LOC = (max_ac + min_ac) / 2
        _AC_SCALE = (max_ac - min_ac) / 2

        for t in buffer:
            t["actions"] = _normalize_actions(t["actions"])
        print("built task", task, "with rollout steps", _MAX_STEPS)
        return buffer


class RoboSuiteTask(SimTask):
    def build_eval_env(self, task, n_cams, obs_dim, ac_dim):
        assert n_cams == 1, "Only support single cam tasks!"
        if task == "transport":
            assert obs_dim == 22, "Robosuite obs_dim should be 22!"
            assert ac_dim == 14, "Robosuite should have ac_dim of 14!"
        else:
            assert obs_dim == 11, "Robosuite obs_dim should be 11!"
            assert ac_dim == 7, "Robosuite should have ac_dim of 7!"
        self.eval_env = _make_env(task)

    def rollout_sim(self, agent, device_id, frame_buffer):
        env = self.eval_env
        transform = self.test_transform

        success_flags, total_rewards = [], []
        for i in range(_N_ROLLOUTS):
            print(f"Rollout {i}", end="\r")
            o = env.reset()
            done = False
            t = 0
            total_reward = 0
            act_history = None

            while not done and t < _MAX_STEPS and not env.is_success()["task"]:
                o = torch.from_numpy(_obs_dict_to_vec(o))[None].to(device_id)
                raw_img = _render(env)
                frame_buffer.append(raw_img)
                i = (
                    torch.from_numpy(raw_img).permute((2, 0, 1)).float().to(device_id)
                    / 255
                )
                i = dict(cam0=transform(i)[None][None])

                with torch.no_grad():
                    acs = agent.get_actions(i, o)[0]

                acs = acs.cpu().numpy()
                if act_history is None:
                    act_history = deque(maxlen=len(acs))
                act_history.append(acs)

                num_actions = len(act_history)
                curr_act_preds = np.stack(
                    [
                        pred_actions[i]
                        for (i, pred_actions) in zip(
                            range(num_actions - 1, -1, -1), act_history
                        )
                    ]
                )

                # compute the weighted average across all predictions for this timestep
                weights = np.exp(-EXP_WEIGHT * np.arange(num_actions))[::-1]
                weights = weights / weights.sum()
                ac = np.sum(weights[:, None] * curr_act_preds, axis=0)

                # denormalize then execute on robot
                ac = _denormalize_action(ac)
                o, r, done, _ = env.step(ac)

                # process the env return and break if done
                t += 1
                total_reward += r
                if done:
                    break

                # for ac_step in acs:
                #     o, r, done, _ = env.step(ac_step)
                #     t += 1; total_reward += r
                #     if done:
                #         break
            success_flags.append(float(env.is_success()["task"]))
            total_rewards.append(float(total_reward))
        return np.mean(success_flags), np.mean(total_rewards)
