# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, IterableDataset

from data4robotics.replay_buffer import IterableWrapper

_TEST_WORKERS = 4


def _build_data_loader(buffer, batch_size, num_workers, is_train=False):
    if is_train and not isinstance(buffer, IterableDataset):
        buffer = IterableWrapper(buffer)

    return DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not isinstance(buffer, IterableDataset),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=lambda _: np.random.seed(),
    )


class DefaultTask:
    def __init__(
        self,
        train_buffer,
        test_buffer,
        n_cams,
        obs_dim,
        ac_dim,
        batch_size,
        num_workers,
    ):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(
            train_buffer, batch_size, num_workers, is_train=True
        )
        if test_buffer is not None:
            test_workers = min(num_workers, _TEST_WORKERS)
            self.test_loader = _build_data_loader(test_buffer, batch_size, test_workers)

    def eval(self, trainer, global_step):
        losses = []
        for batch in self.test_loader:
            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

        mean_val_loss = np.mean(losses)
        print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"eval/task_loss": mean_val_loss}, step=global_step)


class BCTask(DefaultTask):
    def eval(self, trainer, global_step):
        losses = []
        action_l2, action_lsig = [], []
        for batch in self.test_loader:
            (imgs, obs), actions, mask = batch
            imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
            obs, actions, mask = [
                ar.to(trainer.device_id) for ar in (obs, actions, mask)
            ]

            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

                # compare predicted actions versus GT
                pred_actions = trainer.model.get_actions(imgs, obs)

                # calculate l2 loss between pred_action and action
                l2_delta = torch.square(mask * (pred_actions - actions))
                l2_delta = l2_delta.sum((1, 2)) / mask.sum((1, 2))

                # calculate the % of time the signs agree
                lsig = torch.logical_or(
                    torch.logical_and(actions > 0, pred_actions <= 0),
                    torch.logical_and(actions <= 0, pred_actions > 0),
                )
                lsig = (lsig.float() * mask).sum((1, 2)) / mask.sum((1, 2))

                # log mean error values
                action_l2.append(l2_delta.mean().item())
                action_lsig.append(lsig.mean().item())

        mean_val_loss = np.mean(losses)
        ac_l2, ac_lsig = np.mean(action_l2), np.mean(action_lsig)
        print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
        print(f"Step: {global_step}\tAC L2={ac_l2:.2f}\tAC LSig={ac_lsig:.2f}")

        if wandb.run is not None:
            wandb.log(
                {
                    "eval/task_loss": mean_val_loss,
                    "eval/action_l2": ac_l2,
                    "eval/action_lsig": ac_lsig,
                },
                step=global_step,
            )
