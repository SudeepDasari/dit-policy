# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from data4robotics import load_resnet18, load_vit

# load strongest vit/resnet models
vit_transform, vit_model = load_vit()
res_transform, res_model = load_resnet18()


# get embeddings from each network
input_img = torch.rand((1, 3, 480, 640)).cuda()
emb_vit = vit_model(vit_transform(input_img))
emb_res = res_model(res_transform(input_img))


# print out shapes
print("vit_base embedding shape:", emb_vit.shape)
print("resnet18 embedding shape:", emb_res.shape)
