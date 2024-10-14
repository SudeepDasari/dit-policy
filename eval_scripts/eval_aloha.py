import argparse
import json
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

import hydra


# speed up torch agent using tensorcores
torch.set_float32_matmul_precision('high')


# add core aloha files to path then do aloha imports
BASE_PATH = os.path.expanduser("~/aloha/")
sys.path.append(BASE_PATH + "src")
sys.path.append(BASE_PATH + "src/aloha_pro/aloha_scripts/")
from aloha_pro.aloha_scripts.real_env import make_real_env


class Policy:
    def __init__(self, agent_path, model_name, args):
        self.args = args

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
        self.agent = torch.compile(agent.eval().cuda().get_actions)

        self.transform = hydra.utils.instantiate(obs_config["transform"])
        self.img_keys = obs_config["imgs"]

        if args.goal_path:
            bgr_img = cv2.imread(args.goal_path)[:, :, :3]
            bgr_img = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_AREA)
            rgb_img = (
                torch.from_numpy(bgr_img[:, :, ::-1].copy()).float().permute((2, 0, 1))
                / 255
            )
            self.goal_img = self.transform(rgb_img)[None].cuda()
        else:
            self.goal_img = None

        print(f"loaded agent from {agent_path}, at step: {save_dict['global_step']}")
        self.temp_ensemble = args.temp_ensemble
        self.pred_horizon = args.pred_horizon
        self.reset()

    def reset(self):
        self.act_history = deque(maxlen=self.pred_horizon)
        self.last_ac = None
        self._last_time = None

    def _proc_images(self, img_dict, size=(256, 256)):
        torch_imgs = dict()
        for i, k in enumerate(self.img_keys):
            bgr_img = img_dict[k][:, :, :3]
            bgr_img = cv2.resize(bgr_img, size, interpolation=cv2.INTER_AREA)
            rgb_img = bgr_img[:, :, ::-1].copy()
            rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
            rgb_img = self.transform(rgb_img)[None].cuda()

            if self.goal_img is not None:
                if k == "cam_high":
                    torch_imgs[f"cam{i}"] = torch.cat((self.goal_img, rgb_img), 0)[None]
                else:
                    zero_pad = torch.zeros_like(self.goal_img)
                    torch_imgs[f"cam{i}"] = torch.cat((zero_pad, rgb_img), 0)[None]
            else:
                torch_imgs[f"cam{i}"] = rgb_img[None]

        return torch_imgs

    def _proc_state(self, qpos):
        return torch.from_numpy(qpos).float()[None].cuda()

    def _infer_policy(self, obs):
        start = time.time()
        img = self._proc_images(obs["images"])
        print("Image processing time:", time.time() - start)
        start = time.time()
        state = self._proc_state(obs["qpos"])
        print("State processing time:", time.time() - start)

        start = time.time()
        with torch.no_grad():
            ac = self.agent(img, state)
            ac = ac[0].cpu().numpy().astype(np.float32)[: self.args.pred_horizon]
        print("Inference time:", time.time() - start)

        # make sure the model predicted enough steps
        assert (
            len(ac) >= self.args.pred_horizon
        ), "model did not return enough predictions!"
        return ac

    def _forward_ensemble(self, obs):
        ac = self._infer_policy(obs)
        self.act_history.append(ac)

        # potentially consider not ensembling every timestep.

        # handle temporal blending
        num_actions = len(self.act_history)
        print("Num actions:", num_actions)
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.args.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()

        # return the weighted average across all predictions for this timestep
        return np.sum(weights[:, None] * curr_act_preds, axis=0)

    def _forward_chunked(self, obs):
        if not len(self.act_history):
            acs = self._infer_policy(obs)
            for ac in acs:
                self.act_history.append(ac)

        raw_ac = self.act_history.popleft()
        last_ac = self.last_ac if self.last_ac is not None else raw_ac
        self.last_ac = self.args.gamma * raw_ac + (1 - self.args.gamma) * last_ac
        return self.last_ac.copy()

    def forward(self, obs):
        ac = (
            self._forward_ensemble(obs)
            if self.temp_ensemble
            else self._forward_chunked(obs)
        )

        # denormalize the actions
        ac = ac * self.scale + self.loc

        # check effective HZ
        if self._last_time is not None:
            delta = time.time() - self._last_time
            if delta < self.args.period:
                time.sleep(self.args.period - delta)
            print("Effective HZ:", 1.0 / (time.time() - self._last_time))
        self._last_time = time.time()
        return ac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--T", default=400, type=int)
    parser.add_argument("--temp_ensemble", default=False, action="store_true")
    parser.add_argument("--num_rollouts", default=1, type=int)
    parser.add_argument("--pred_horizon", default=48, type=int)
    parser.add_argument("--exp_weight", default=0, type=float)
    parser.add_argument("--hz", default=48, type=float)
    parser.add_argument("--gamma", default=0.85, type=float)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--goal_path", default=None, type=str)

    args = parser.parse_args()
    args.period = 1.0 / args.hz

    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split("/")[-1]
    policy = Policy(agent_path, model_name, args)

    env = make_real_env(init_node=True)

    if args.save_dir:
        next_highest = get_highest_rollout_num(args.save_dir) + 1
    else:
        # Default to starting at 0
        next_highest = 0

    # Roll out the policy num_rollout times
    for rollout_num in range(next_highest, args.num_rollouts + next_highest):

        last_input = None
        while last_input != "y":
            if last_input == "r":
                obs = env.reset()
            last_input = input("Continue with rollout (y; r to reset now)?")

        policy.reset()

        obs_data = []

        obs = env.reset()
        start_time = time.time()
        obs_data.append(obs)

        for _ in range(args.T):
            ac = policy.forward(obs.observation)
            obs = env.step(ac)
            obs_data.append(obs)

        end_time = time.time()

        if args.save_dir:
            # Save the rollout video if save_dir is provided

            rollout_name = f"episode_{rollout_num}.mp4"
            save_path = os.path.join(args.save_dir, rollout_name)
            save_thread = threading.Thread(
                target=save_rollout_video,
                args=(obs_data, save_path, policy.img_keys, end_time - start_time),
            )
            save_thread.start()

        env._reset_gripper()


def get_highest_rollout_num(save_dir):
    """
    Get the highest rollout number in the save directory
    """
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist.")

    files = [
        os.path.basename(f)
        for f in os.listdir(save_dir)
        if os.path.isfile(os.path.join(save_dir, f))
    ]
    if not files:
        return -1  # No files yet
    return max([int(re.search(r"\d+", f_name)[0]) for f_name in files])


def save_rollout_video(obs, path, camera_names, length_of_episode):
    """
    Save the policy rollout to a video
    """
    t0 = time.time()

    # Get the list
    image_dict = {}

    for cam_name in camera_names:
        image_dict[cam_name] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while len(obs) > 1:
        ts = obs.pop(0)
        for cam_name in camera_names:
            image_dict[cam_name].append(ts.observation["images"][cam_name])

    cam_names = list(image_dict.keys())

    all_cam_videos = []
    for cam_name in cam_names:
        all_cam_videos.append(image_dict[cam_name])
    all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

    n_frames, h, w, _ = all_cam_videos.shape
    fps = int(
        n_frames / length_of_episode
    )  # This is an estimate, but the frames are not uniformly distributed when rolling out the policy, leading to skips in the video.

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(n_frames):
        image = all_cam_videos[t]
        image = image[:, :, [0, 1, 2]]
        out.write(image)
    out.release()

    print(f"Saving: {time.time() - t0:.1f} secs")


if __name__ == "__main__":
    main()
