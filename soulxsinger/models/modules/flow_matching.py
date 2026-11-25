# https://github.com/open-mmlab/Amphion/blob/main/models/svc/flow_matching_transformer/fmt_model.py

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .llama import DiffLlama


class FlowMatchingTransformer(nn.Module):
    """Inference-only flow matching with continuous SVC conditions."""

    def __init__(self, config):
        super().__init__()
        self.mel_dim = config.mel_dim
        self.hidden_size = config.hidden_size
        self.cond_emb = nn.Linear(config.cond_codebook_size, self.hidden_size)
        self.reset_parameters()
        self.diff_estimator = DiffLlama(
            mel_dim=self.mel_dim,
            hidden_size=self.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
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
        h = 1.0 / n_timesteps
        prompt_len = prompt.shape[1]
        target_len = cond.shape[1] - prompt_len

        if x_mask == None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)  # (B, T)
        if prompt_mask == None:
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
                rescale_flow_pred = (
                    flow_pred_cfg * pos_flow_pred_std / flow_pred_cfg.std()
                )
                flow_pred = (
                    rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg
                )

            dxt = flow_pred * h
            xt = xt + dxt

        return xt
