import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention as slow_attn

from grn.models.rope import apply_rotary_emb

try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
except ImportError:
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))

    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)

    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12, num_key_value_heads=-1,
        use_flex_attn=False, qwen_qkvo_bias=False, **kwargs,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_key_value_heads == -1 or num_heads % num_key_value_heads == 0

        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else num_heads
        self.q_proj = nn.Linear(embed_dim, self.num_heads*self.head_dim, bias=qwen_qkvo_bias)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, embed_dim, bias=qwen_qkvo_bias)
        self.q_norm = FastRMSNorm(self.head_dim)
        self.k_norm = FastRMSNorm(self.head_dim)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.caching = False
        self.cached_k = {}
        self.cached_v = {}
        self.cached_split_cond_uncond = {}

        self.use_flex_attn = use_flex_attn

    def kv_caching(self, enable: bool):
        self.caching = enable
        self.cached_k = {}
        self.cached_v = {}
        self.cached_split_cond_uncond = {}


    def forward(self, x, cu_seqlens, max_seqlen, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, rope2d_freqs_grid=[], scale_ind=0, context_info=None, last_diffusion_step=True, ref_text_scale_inds=[], use_cfg=False, split_cond_uncond=[], **kwargs):

        B, L, C = x.shape
        hidden_states = x
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).contiguous()
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).contiguous()
        value_states = self.v_proj(hidden_states).view(hidden_shape).contiguous()


        query_states, key_states = apply_rotary_emb(query_states, key_states, rope2d_freqs_grid)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attn_bias_or_two_vector is None:

            from flash_attn import flash_attn_varlen_func
            attn_output = flash_attn_varlen_func(
                q = query_states.squeeze(0),
                k = key_states.squeeze(0),
                v = value_states.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scale,
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            attn_output = attn_output.reshape(B, L, C).contiguous()
        else:

            attn_output = slow_attn(query=query_states.transpose(1, 2), key=key_states.transpose(1, 2), value=value_states.transpose(1, 2), scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)


        attn_output = self.o_proj(attn_output)

        return attn_output

class SelfAttnBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_key_value_heads, mlp_ratio=4.,
        use_flex_attn=False,
        qwen_qkvo_bias=False, use_ada_layer_norm=False, **kwargs,
    ):
        super(SelfAttnBlock, self).__init__()
        self.C = embed_dim
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_key_value_heads=num_key_value_heads,
            use_flex_attn=use_flex_attn, qwen_qkvo_bias=qwen_qkvo_bias, **kwargs,
        )
        self.mlp = Qwen3MLP(hidden_size=embed_dim, intermediate_size=round(embed_dim * mlp_ratio / 256) * 256)
        self.use_ada_layer_norm = use_ada_layer_norm
        if self.use_ada_layer_norm:
            self.modulation = nn.Parameter(torch.randn(1, 6, embed_dim) / embed_dim**0.5)
            self.input_layernorm = WanLayerNorm(embed_dim)
            self.post_attention_layernorm = WanLayerNorm(embed_dim)
        else:
            self.input_layernorm = FastRMSNorm(embed_dim)
            self.post_attention_layernorm = FastRMSNorm(embed_dim)


    def forward(self, x, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector, attn_fn=None, rope2d_freqs_grid=[], scale_ind=0, context_info=None, last_diffusion_step=True, ref_text_scale_inds=[], use_cfg=False, split_cond_uncond=[], **kwargs):


        if self.use_ada_layer_norm:
            assert e0.dtype == torch.float32
            e = e0
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            residual = x
            hidden_states = x
            hidden_states = self.input_layernorm(hidden_states).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
            hidden_states = self.attn(hidden_states, cu_seqlens, max_seqlen, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_ind, context_info, last_diffusion_step, ref_text_scale_inds, use_cfg, split_cond_uncond, **kwargs)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                hidden_states = residual + hidden_states * e[2].squeeze(2)

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
            hidden_states = self.mlp(hidden_states)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                hidden_states = residual + hidden_states * e[5].squeeze(2)
        else:
            residual = x
            hidden_states = x
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.attn(hidden_states, cu_seqlens, max_seqlen, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_ind, context_info, last_diffusion_step, ref_text_scale_inds, use_cfg, split_cond_uncond, **kwargs)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        return hidden_states
