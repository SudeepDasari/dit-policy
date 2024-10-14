# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, model, restore_path):
        super().__init__()
        self._model = model
        if restore_path:
            print("Restoring model from", restore_path)
            state_dict = torch.load(restore_path, map_location="cpu")
            state_dict = (
                state_dict["features"]
                if "features" in state_dict
                else state_dict["model"]
            )
            self.load_state_dict(state_dict)

    @property
    def embed_dim(self):
        raise NotImplementedError
