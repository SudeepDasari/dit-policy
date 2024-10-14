# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import torch
from torch import nn


class _BatchNorm1DHelper(nn.BatchNorm1d):
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        return super().forward(x)


class BaseAgent(nn.Module):
    def __init__(
        self,
        odim,
        features,
        n_cams,
        imgs_per_cam,
        use_obs=False,
        share_cam_features=False,
        early_fusion=False,
        dropout=0,
        feat_norm=None,
        token_dim=None,
    ):
        super().__init__()

        # store visual features (duplicate weights if shared)
        self._share_cam_features = share_cam_features
        if self._share_cam_features:
            self.visual_features = features
        else:
            feat_list = [features] + [copy.deepcopy(features) for _ in range(1, n_cams)]
            self.visual_features = nn.ModuleList(feat_list)

        self.early_fusion = early_fusion
        imgs_per_cam = 1 if early_fusion else imgs_per_cam
        self._token_dim = features.embed_dim
        self._n_tokens = imgs_per_cam * n_cams * features.n_tokens

        # handle obs tokenization strategies
        if use_obs == "add_token":
            self._obs_strat = "add_token"
            self._n_tokens += 1
            self._obs_proc = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(odim, self._token_dim)
            )
        elif use_obs == "pad_img_tokens":
            self._obs_strat = "pad_img_tokens"
            self._token_dim += odim
            self._obs_proc = nn.Dropout(p=0.2)
        else:
            assert not use_obs
            self._obs_strat = None

        # build (optional) token feature projection layer
        linear_proj = nn.Identity()
        if token_dim is not None and token_dim != self._token_dim:
            linear_proj = nn.Linear(self._token_dim, token_dim)
            self._token_dim = token_dim

        # build feature normalization layers
        if feat_norm == "batch_norm":
            norm = _BatchNorm1DHelper(self._token_dim)
        elif feat_norm == "layer_norm":
            norm = nn.LayerNorm(self._token_dim)
        else:
            assert feat_norm is None
            norm = nn.Identity()

        # final token post proc network
        self.post_proc = nn.Sequential(linear_proj, norm, nn.Dropout(dropout))

    def forward(self, imgs, obs, ac_flat, mask_flat):
        raise NotImplementedError

    def get_actions(self, img, obs):
        raise NotImplementedError

    def tokenize_obs(self, imgs, obs, flatten=False):
        # start by getting image tokens
        tokens = self.embed(imgs)

        if self._obs_strat == "add_token":
            obs_token = self._obs_proc(obs)[:, None]
            tokens = torch.cat((tokens, obs_token), 1)
        elif self._obs_strat == "pad_img_tokens":
            obs = self._obs_proc(obs)
            obs = obs[:, None].repeat((1, tokens.shape[1], 1))
            tokens = torch.cat((obs, tokens), 2)
        else:
            assert self._obs_strat is None

        tokens = self.post_proc(tokens)
        if flatten:
            return tokens.reshape((tokens.shape[0], -1))
        return tokens

    def embed(self, imgs):
        def embed_helper(net, im):
            if self.early_fusion and len(im.shape) == 5:
                T = im.shape[1]
                im = torch.cat([im[:, t] for t in range(T)], 1)
                return net(im)
            elif len(im.shape) == 5:
                B, T, C, H, W = im.shape
                embeds = net(im.reshape((B * T, C, H, W)))
                embeds = embeds.reshape((B, -1, net.embed_dim))
                return embeds

            assert len(im.shape) == 4
            return net(im)

        if self._share_cam_features:
            embeds = [
                embed_helper(self.visual_features, imgs[f"cam{i}"])
                for i in range(self._n_cams)
            ]
        else:
            embeds = [
                embed_helper(net, imgs[f"cam{i}"])
                for i, net in enumerate(self.visual_features)
            ]
        return torch.cat(embeds, dim=1)

    @property
    def n_cams(self):
        return self._n_cams

    @property
    def ac_chunk(self):
        raise NotImplementedError

    @property
    def token_dim(self):
        return self._token_dim

    @property
    def n_tokens(self):
        return self._n_tokens


class MLPAgent(BaseAgent):
    def __init__(
        self,
        features,
        policy,
        shared_mlp,
        odim,
        n_cams,
        use_obs,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        early_fusion=False,
        feat_norm="layer_norm",
        token_dim=None,
    ):

        # initialize obs and img tokenizers
        super().__init__(
            odim=odim,
            features=features,
            n_cams=n_cams,
            imgs_per_cam=imgs_per_cam,
            use_obs=use_obs,
            share_cam_features=share_cam_features,
            early_fusion=early_fusion,
            dropout=dropout,
            feat_norm=feat_norm,
            token_dim=token_dim,
        )

        # assign policy class
        self._policy = policy

        mlp_in = self.n_tokens * self.token_dim
        mlp_def = [mlp_in] + shared_mlp
        layers = []
        for i, o in zip(mlp_def[:-1], mlp_def[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self._mlp = nn.Sequential(*layers)

    def forward(self, imgs, obs, ac_flat, mask_flat):
        s_t = self._mlp_forward(imgs, obs)
        action_dist = self._policy(s_t)
        loss = (
            -torch.mean(action_dist.masked_log_prob(ac_flat, mask_flat))
            if hasattr(action_dist, "masked_log_prob")
            else -(action_dist.log_prob(ac_flat) * mask_flat).sum() / mask_flat.sum()
        )
        return loss

    def get_actions(self, img, obs, zero_std=True):
        policy_in = self._mlp_forward(img, obs)
        return self._policy.get_actions(policy_in, zero_std=zero_std)

    def _mlp_forward(self, imgs, obs):
        tokens_flat = self.tokenize_obs(imgs, obs, flatten=True)
        return self._mlp(tokens_flat)

    @property
    def ac_chunk(self):
        return self._policy.ac_chunk
