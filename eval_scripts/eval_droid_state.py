import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# droid robot imports
from droid.robot_env import RobotEnv
from droid.user_interface.eval_gui import EvalGUI
from scipy.spatial.transform import Rotation as R

import hydra

PRED_HORIZON = 16
EXP_WEIGHT = 0
GRIP_THRESH = 0.55


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    return vec / norm


def rot6d_to_euler(d6):
    a1, a2 = d6[:3], d6[3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return rmat_to_euler(out)


class DROIDStatePolicy:
    def __init__(self, agent_path, model_name):
        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "obs_config.yaml"), "r") as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "ac_norm.json"), "r") as f:
            ac_norm_dict = json.load(f)
            loc, scale = ac_norm_dict["loc"], ac_norm_dict["scale"]
            self.loc = np.array(loc).astype(np.float32)
            self.scale = np.array(scale).astype(np.float32)

        agent = hydra.utils.instantiate(agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        agent.load_state_dict(save_dict["model"])
        self.agent = agent.eval().cuda()

        self.transform = hydra.utils.instantiate(obs_config["transform"])
        self.img_key = obs_config["img"]

        print(f"loaded agent from {agent_path}, at step: {save_dict['global_step']}")
        self.act_history = deque(maxlen=PRED_HORIZON)
        self._last_time = None

    def _proc_image(self, zed_img, size=(256, 256)):
        bgr_img = zed_img[:, :, :3]
        bgr_img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
        rgb_img = bgr_img[:, :, ::-1].copy()
        rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
        return {"cam0": self.transform(rgb_img)[None].cuda()}

    def _proc_state(self, cart_pos, grip_pos):
        state = np.concatenate((cart_pos, np.array([grip_pos]))).astype(np.float32)
        return torch.from_numpy(state)[None].cuda()

    def forward(self, obs):
        img = self._proc_image(obs["image"][self.img_key])
        state = self._proc_state(
            obs["robot_state"]["cartesian_position"],
            obs["robot_state"]["gripper_position"],
        )

        with torch.no_grad():
            ac = self.agent.get_actions(img, state)
            ac = ac[0].cpu().numpy().astype(np.float32)[:PRED_HORIZON]
            self.act_history.append(ac)

        # handle temporal blending
        num_actions = len(self.act_history)
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-EXP_WEIGHT * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        ac = np.sum(weights[:, None] * curr_act_preds, axis=0)

        # denormalize the actions and swap to R6
        ac = ac * self.scale + self.loc
        if len(ac) == 10:
            xyz, r6, grip = ac[:3], ac[3:9], ac[9:]
            ac = np.concatenate((xyz, rot6d_to_euler(r6), grip))
        assert len(ac) == 7, "Assuming 7d action dim!"

        # threshold the gripper to make crisp grasp decisions
        if ac[-1] > GRIP_THRESH:
            ac[-1] = 1.0

        print("current", obs["robot_state"]["cartesian_position"])
        print("action", ac)
        cur_time = time.time()
        if self._last_time is not None:
            print("Effective HZ:", 1.0 / (cur_time - self._last_time))
        self._last_time = cur_time
        print()
        return ac

    def load_goal_imgs(self, goal_dict):
        pass

    def load_lang(self, text):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split("/")[-1]
    policy = DROIDStatePolicy(agent_path, model_name)

    # start up DROID eval gui
    env = RobotEnv(action_space="cartesian_position")
    EvalGUI(policy=policy, env=env)


if __name__ == "__main__":
    main()
