# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from data4robotics.agent import BaseAgent


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        super().__init__()

        # positional encoding blocks
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # encoder blocks
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)
    
    def forward_enc(self, obs_enc):
        obs_enc = obs_enc.transpose(0, 1)
        pos = self.enc_pos(obs_enc)
        enc_cache = self.encoder(obs_enc, pos)
        return enc_cache

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)
        
        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        dec_in = ac_tokens + self.dec_pos

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, enc_cache)

        # apply final epsilon prediction layer
        return self.eps_out(dec_out, time_enc, enc_cache[-1])


class DiffusionTransformerAgent(BaseAgent):
    def __init__(
        self,
        features,
        odim,
        n_cams,
        use_obs,
        ac_dim,
        ac_chunk,
        train_diffusion_steps,
        eval_diffusion_steps,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        early_fusion=False,
        feat_norm=None,
        token_dim=None,
        noise_net_kwargs=dict(),
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

        self.noise_net = _DiTNoiseNet(
            ac_dim=ac_dim,
            ac_chunk=ac_chunk,
            **noise_net_kwargs,
        )
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk

        assert (
            eval_diffusion_steps <= train_diffusion_steps
        ), "Can't eval with more steps!"
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def forward(self, imgs, obs, ac_flat, mask_flat):
        # get observation encoding and sample noise/timesteps
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)
        timesteps = torch.randint(
            low=0, high=self._train_diffusion_steps, size=(B,), device=device
        ).long()

        # diffusion unet logic assumes [B, T, adim]
        mask = mask_flat.reshape((B, self.ac_chunk, self.ac_dim))
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        noise = torch.randn_like(actions)

        # construct noise actions given real actions, noise, and diffusion schedule
        noise_acs = self.diffusion_schedule.add_noise(actions, noise, timesteps)
        _, noise_pred = self.noise_net(noise_acs, timesteps, s_t)

        # calculate loss for noise net
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        loss = (loss * mask).sum(1)  # mask the loss to only consider "real" acs
        return loss.mean()

    def get_actions(self, imgs, obs, n_steps=None):
        # get observation encoding and sample noise
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)
        enc_cache = None
        noise_actions = torch.randn(B, self.ac_chunk, self.ac_dim, device=device)

        # set number of steps
        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert (
                n_steps <= self._train_diffusion_steps
            ), f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps

        enc_cache = self.noise_net.forward_enc(s_t)

        # begin diffusion process
        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = (
            self.diffusion_schedule.alphas_cumprod.to(device)
        )
        for timestep in self.diffusion_schedule.timesteps:
            # predict noise given timestep
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net.forward_dec(noise_actions, batched_timestep, enc_cache)

            # take diffusion step
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=timestep, sample=noise_actions
            ).prev_sample

        # return final action post diffusion
        return noise_actions

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim
