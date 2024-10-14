# Copyright (c) Sudeep Dasari, 2023
# schedule_builder inspired from Diffusion Policy Codebase (Chi et al; arXiv:2303.04137)

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools

import torch.optim as optim
from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType
from torch.optim import lr_scheduler


def optim_builder(optimizer_type, optimizer_kwargs):
    optimizer_class = getattr(optim, optimizer_type)
    return functools.partial(optimizer_class, **optimizer_kwargs)


def schedule_builder(schedule_type, schedule_kwargs, from_diffusers=False):
    if from_diffusers:
        schedule_type = SchedulerType(schedule_type)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[schedule_type]

        if schedule_type == SchedulerType.CONSTANT:
            return functools.partial(schedule_func, **schedule_kwargs)

        assert (
            "num_warmup_steps" in schedule_kwargs
        ), "Scheduler requires num_warmup_steps!"
        if schedule_type == SchedulerType.CONSTANT_WITH_WARMUP:
            return functools.partial(schedule_func, **schedule_kwargs)

        # All other schedulers require `num_training_steps`
        assert (
            "num_training_steps" in schedule_kwargs
        ), "Scheduler requires num_training_steps!"
        return functools.partial(schedule_func, **schedule_kwargs)
    schedule_class = getattr(lr_scheduler, schedule_type)
    return functools.partial(schedule_class, **schedule_kwargs)
