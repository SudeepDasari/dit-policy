import argparse
from pathlib import Path
import time
import cv2
import hydra
import numpy as np
import torch
import yaml
import os
import json
import random
import pickle as pkl
from robobuf import ReplayBuffer as RB
from data4robotics.transforms import get_transform_by_name


# constants for data loading
BUF_SHUFFLE_RNG = 3904767649 # from replay_buffer.py
n_test_trans = 500 # usually hardocded in task/franka.yaml


class BaselinePolicy:
    def __init__(self, agent_path, model_name):
        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "exp_config.yaml"), "r") as f:
            config_yaml = f.read()
            exp_config = yaml.safe_load(config_yaml)
            self.cam_idx = exp_config['params']['task']['train_buffer']['cam_idx']

        agent = hydra.utils.instantiate(agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        agent.load_state_dict(save_dict['model'])
        self.agent = agent.eval().cuda()

        self.transform = get_transform_by_name('preproc')
    
    def _proc_image(self, rgb_img, size=(256,256)):
        rgb_img = cv2.resize(rgb_img, size, interpolation=cv2.INTER_AREA)
        rgb_img = torch.from_numpy(rgb_img).float().permute((2, 0, 1)) / 255
        return self.transform(rgb_img)[None].cuda()
    
    def forward(self, img, obs):
        img = self._proc_image(img)
        state = torch.from_numpy(obs)[None].float().cuda()

        with torch.no_grad():
            ac = self.agent.get_actions(img, state)
        ac = ac[0].cpu().numpy().astype(np.float32)
        return ac
    
    @property
    def ac_chunk(self):
        return self.agent.ac_chunk


def _get_data(idx, buf, ac_chunk, cam_idx):
    t = buf[idx]
    loop_t, chunked_actions = t, []
    for _ in range(ac_chunk):
        if loop_t.next is None:
            break
        chunked_actions.append(loop_t.action[None])
        loop_t = loop_t.next

    if len(chunked_actions) != ac_chunk:
        raise ValueError

    i_t, o_t = t.obs.image(cam_idx), t.obs.state
    i_t_prime, o_t_prime = t.next.obs.image(cam_idx), t.next.obs.state
    a_t = np.concatenate(chunked_actions, 0)
    return i_t, o_t, a_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--buffer_path", default='/scratch/sudeep/toaster3/buf.pkl')
    args = parser.parse_args()

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split('/')[-1]
    policy = BaselinePolicy(agent_path, model_name)

    # build data loader
    cam_idx = policy.cam_idx
    print('cam_idx:', cam_idx)
    with open(args.buffer_path, 'rb') as f:
        buf = RB.load_traj_list(pkl.load(f))

    # shuffle the list with the fixed seed
    rng = random.Random(BUF_SHUFFLE_RNG)

    # get and shuffle list of buf indices, and get test data
    index_list = list(range(len(buf)))
    rng.shuffle(index_list)
    index_list = index_list[:n_test_trans]

    l2s, lsigs = [], []
    for idx in index_list[:50]:
        i_t, o_t, a_t = _get_data(idx, buf, policy.ac_chunk, cam_idx)
        pred_ac = policy.forward(i_t, o_t)
        
        # calculate deltas
        l2 = np.linalg.norm(a_t - pred_ac)
        lsign = np.sum(np.logical_or(np.logical_and(a_t > 0, pred_ac <= 0),
                                     np.logical_and(a_t <= 0, pred_ac > 0)))
        l2s.append(l2); lsigs.append(lsign)

        print('\n')
        print('a_t', a_t)
        print('pred_ac', pred_ac)
        print(f'losses: l2={l2:0.2f}\tlsign={lsign}')
        print('\n')

    print(f'avg losses: l2={np.mean(l2s):0.3f}\tlsign={np.mean(lsigs):0.3f}')

if __name__ == "__main__":
    main()
