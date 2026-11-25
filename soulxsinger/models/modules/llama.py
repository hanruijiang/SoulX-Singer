from transformers import LlamaConfig, LlamaModel
import torch
import torch.nn as nn
import math

from transformers.models.llama.modeling_llama import LlamaDecoderLayer


# sinusoidal positional encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True  # disable automatic init

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        """Override to adaptive layer norm"""
        super().__init__(config, layer_idx)  # init attention, mlp, etc.
        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dim_cond=config.hidden_size
        )

    def forward(self, hidden_states, cond_embedding, attention_mask, position_ids):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, cond_embedding=cond_embedding)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, cond_embedding=cond_embedding)
        return residual + self.mlp(hidden_states)


class DiffLlama(LlamaModel):
    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        config=LlamaConfig(0, 256, 1024, 1, 1),
    ):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    LlamaConfig(
                        hidden_size=hidden_size,
                        num_attention_heads=num_heads,
                        max_position_embeddings=4096,
                        intermediate_size=hidden_size * 4,
                    ),
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_mlp = nn.Sequential(
            nn.Linear(mel_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, mel_dim),
        )

        self.embed_tokens = None

        self.post_init()

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, inputs_embeds):
        # Non-causal attention over the complete sequence.
        bsz, src_len = attention_mask.shape
        tgt_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        expanded_mask = attention_mask[:, None, None, :].expand(
            bsz, 1, tgt_len, src_len
        ).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        ).to(inputs_embeds.device)

    def forward(self, x, diffusion_step, cond, x_mask):
        _, seq_length, _ = x.shape
        cond_embedding = self.cond_mlp(cond)
        x = self.mel_mlp(x)
        diffusion_step = self.diff_step_embedding(diffusion_step).to(x.device)
        diffusion_step = self.diff_step_mlp(diffusion_step)
        hidden_states = x + cond_embedding

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        attention_mask = self._prepare_decoder_attention_mask(x_mask, hidden_states)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, cond_embedding=diffusion_step,
                attention_mask=attention_mask, position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step)
        return self.mel_out_mlp(hidden_states)
