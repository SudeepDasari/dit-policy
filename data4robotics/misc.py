# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools
import os
import signal
import sys

import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from data4robotics.transforms import get_transform_by_name

OmegaConf.register_new_resolver("env", lambda x: os.environ[x])
OmegaConf.register_new_resolver(
    "base", lambda: os.path.dirname(os.path.abspath(__file__))
)
OmegaConf.register_new_resolver("transform", lambda name: get_transform_by_name(name))
OmegaConf.register_new_resolver("mult", lambda x, y: int(x) * int(y))
OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("index", lambda arr, idx: arr[idx])
OmegaConf.register_new_resolver("len", lambda arr: len(arr))


GLOBAL_STEP = 0
REQUEUE_CAUGHT = False


def _signal_helper(signal, frame, prior_handler, trainer, ckpt_path):
    global REQUEUE_CAUGHT, GLOBAL_STEP
    REQUEUE_CAUGHT = True

    # save train checkpoint
    print(f"Caught requeue signal at step: {GLOBAL_STEP}")
    trainer.save_checkpoint(ckpt_path, GLOBAL_STEP)

    # return back to submitit handler if it exists
    if callable(prior_handler):
        return prior_handler(signal, frame)
    return sys.exit(-1)


def set_checkpoint_handler(trainer, ckpt_path):
    global REQUEUE_CAUGHT
    REQUEUE_CAUGHT = False
    prior_handler = signal.getsignal(signal.SIGUSR2)
    handler = functools.partial(
        _signal_helper,
        prior_handler=prior_handler,
        trainer=trainer,
        ckpt_path=ckpt_path,
    )
    signal.signal(signal.SIGUSR2, handler)


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    if wandb_cfg.debug:
        return "null_id"
    try:
        job_id = HydraConfig().get().job.num
        override_dirname = HydraConfig().get().job.override_dirname
        name = f"{wandb_cfg.sweep_name_prefix}-{job_id}"
        notes = f"{override_dirname}"
    except:
        name, notes = wandb_cfg.name, None

    wandb_run = wandb.init(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        config=job_config,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )
    return wandb_run.id


def init_job(cfg):
    cfg_yaml = OmegaConf.to_yaml(cfg)
    if os.path.exists("exp_config.yaml"):
        old_config = yaml.safe_load(open("exp_config.yaml", "r"))
        create_wandb_run(cfg.wandb, old_config["params"], old_config["wandb_id"])
        resume_model = cfg.checkpoint_path
        assert os.path.exists(resume_model), "{} does not exist!".format(
            cfg.checkpoint_path
        )
    else:
        params = yaml.safe_load(cfg_yaml)
        wandb_id = create_wandb_run(cfg.wandb, params)
        save_dict = dict(wandb_id=wandb_id, params=params)
        yaml.dump(save_dict, open("exp_config.yaml", "w"))
        resume_model = None
        print("Training w/ Config:")
        print(cfg_yaml)
    return resume_model
