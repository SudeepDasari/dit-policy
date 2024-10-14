# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import random

import cv2
import imageio
import numpy as np
import torch
import tqdm
import wandb
from torch.utils.data import Dataset

from data4robotics.task import DefaultTask

# helper functions
_img_to_tensor = (
    lambda x: torch.from_numpy(x.copy()).permute((0, 3, 1, 2)).float() / 255
)
_to_tensor = lambda x: torch.from_numpy(x).float()
BUF_SHUFFLE_RNG = 3904767649
_VID_FPS = 10


class SimTask(DefaultTask):
    def __init__(
        self,
        train_buffer,
        test_transform,
        task,
        n_cams,
        obs_dim,
        ac_dim,
        batch_size,
        num_workers,
    ):
        self.build_eval_env(task, n_cams, obs_dim, ac_dim)
        self.test_transform = test_transform
        super().__init__(
            train_buffer, None, n_cams, obs_dim, ac_dim, batch_size, num_workers
        )

    def build_eval_env(self, task, n_cams, obs_dim, ac_dim):
        raise NotImplementedError()

    def rollout_sim(self, agent, device_id, frame_buffer):
        raise NotImplementedError()

    def eval(self, trainer, global_step):
        frame_buffer = []
        mean_success, mean_rewards = self.rollout_sim(
            trainer.model, trainer.device_id, frame_buffer
        )

        # load success log if it exists or make a new one
        rollout_logs = dict(success=[], reward=[], step=[])
        if os.path.exists("rollout_log.json"):
            rollout_logs = json.load(open("rollout_log.json", "r"))
        rollout_logs["success"].append(mean_success)
        rollout_logs["reward"].append(mean_rewards)
        rollout_logs["step"].append(global_step)
        with open("rollout_log.json", "w") as f:
            json.dump(rollout_logs, f)
            f.write("\n")

        max_success = max(rollout_logs["success"])
        print(f"Max Success={max_success} @ step={global_step}")

        vid_path = f"rollout_itr_{global_step:07d}.mp4"
        writer = imageio.get_writer(vid_path, fps=_VID_FPS)
        for im in frame_buffer:
            im_out = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
            writer.append_data(im_out)
        writer.close()

        if wandb.run is not None:
            wandb.log(
                {
                    "eval/max_success": max_success,
                    "eval/success": mean_success,
                    "eval/rewards": mean_rewards,
                    "eval/rollout": wandb.Video(vid_path),
                },
                step=global_step,
            )


class SimTaskReplayBuffer(Dataset):
    def __init__(
        self,
        task,
        transform=None,
        n_train_demos=200,
        ac_chunk=1,
        cam_indexes=[0],
        goal_indexes=[],
        past_frames=0,
    ):

        # these tasks don't require conditioning so skip
        assert cam_indexes == [0], "only need 0th cam"
        assert goal_indexes == [], "no need for goal indexes"
        assert past_frames == 0, "past frames should be 0"

        buffer_data = self.load_buffer(task)
        assert len(buffer_data) >= n_train_demos, "Not enough demos!"

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)
        rng.shuffle(buffer_data)

        # take n_train_demos demos for training
        buffer_data = buffer_data[:n_train_demos]

        self.transform = transform
        self.s_a_mask = []
        for traj in tqdm.tqdm(buffer_data):
            imgs, obs, acs = traj["images"], traj["observations"], traj["actions"]
            assert len(obs) == len(acs) and len(acs) == len(
                imgs
            ), "All time dimensions must match!"

            # pad camera dimension if needed
            if len(imgs.shape) == 4:
                imgs = imgs[:, None]

            for t in range(len(imgs) - 1):
                i_t = {f"cam{c}": imgs[t, c][None] for c in range(imgs.shape[1])}
                o_t = obs[t]

                loss_mask = np.ones((ac_chunk,), dtype=np.float32)
                a_t = acs[t : t + ac_chunk]
                assert len(a_t) > 0
                if len(a_t) < ac_chunk:
                    missing = ac_chunk - len(a_t)

                    action_pad = np.zeros((missing, a_t.shape[-1])).astype(np.float32)
                    a_t = np.concatenate((a_t, action_pad), axis=0)
                    loss_mask[-missing:] = 0

                self.s_a_mask.append(((i_t, o_t), a_t, loss_mask))

    def load_buffer(self, buffer_path):
        raise NotImplementedError

    def __len__(self):
        return len(self.s_a_mask)

    def __getitem__(self, idx):
        (i_t, o_t), a_t, loss_mask = self.s_a_mask[idx]

        i_t = {k: _img_to_tensor(v) for k, v in i_t.items()}
        if self.transform is not None:
            i_t = {k: self.transform(v) for k, v in i_t.items()}

        o_t, a_t = _to_tensor(o_t), _to_tensor(a_t)
        loss_mask = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))
        assert (
            loss_mask.shape[0] == a_t.shape[0]
        ), "a_t and mask shape must be ac_chunk!"
        return (i_t, o_t), a_t, loss_mask
