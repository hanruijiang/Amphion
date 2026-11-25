# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
from .llama_nar import DiffLlama


class FlowMatchingTransformer(nn.Module):
    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        cond_codebook_size=1024,
        cond_scale_factor=1,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg is not None:
            mel_dim = getattr(cfg, "mel_dim", mel_dim)
            hidden_size = getattr(cfg, "hidden_size", hidden_size)
            num_layers = getattr(cfg, "num_layers", num_layers)
            num_heads = getattr(cfg, "num_heads", num_heads)
            cond_codebook_size = getattr(cfg, "cond_codebook_size", cond_codebook_size)
            cond_scale_factor = getattr(cfg, "cond_scale_factor", cond_scale_factor)

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cond_codebook_size = cond_codebook_size
        self.cond_scale_factor = cond_scale_factor

        self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)

        if cond_scale_factor != 1:
            self.do_resampling = True
            assert np.log2(cond_scale_factor).is_integer()

            up_layers = []
            for _ in range(int(np.log2(cond_scale_factor))):
                up_layers.extend(
                    [
                        nn.ConvTranspose1d(
                            hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                        ),
                        nn.GELU(),
                    ]
                )
            self.resampling_layers = nn.Sequential(*up_layers)
        else:
            self.do_resampling = False

        self.reset_parameters()

        self.diff_estimator = DiffLlama(
            mel_dim=mel_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )


    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    @torch.no_grad()
    def reverse_diffusion(
        self,
        cond,
        prompt,
        x_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        if n_timesteps < 1:
            raise ValueError("Flow matching steps must be positive")
        h = 1.0 / n_timesteps
        prompt_len = prompt.shape[1]
        target_len = cond.shape[1] - prompt_len
        if target_len <= 0:
            raise ValueError("No target frames remain after the timbre prompt")

        if x_mask is None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)  # (B, T)
        if prompt_mask is None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(
                cond.device
            )  # (B, prompt_len)
        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)
        z = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=cond.dtype,
            device=cond.device,
            requires_grad=False,
        )
        xt = z

        # t from 0 to 1: x0 = z ~ N(0, 1)
        for i in range(n_timesteps):
            xt_input = torch.cat([prompt, xt], dim=1)
            t = (0 + (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            flow_pred = self.diff_estimator(xt_input, t, cond, xt_mask)
            flow_pred = flow_pred[:, prompt_len:, :]

            # cfg
            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(
                    xt, t, torch.zeros_like(cond)[:, : xt.shape[1], :], x_mask
                )
                pos_flow_pred_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                cfg_std = flow_pred_cfg.std()
                rescale_flow_pred = (
                    flow_pred_cfg * pos_flow_pred_std / cfg_std
                    if torch.isfinite(cfg_std) and cfg_std > 0
                    else flow_pred_cfg
                )
                flow_pred = (
                    rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg
                )

            dxt = flow_pred * h
            xt = xt + dxt

        return xt
