import json
import math
import os
import time
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import tqdm
from PIL import Image, ImageDraw
from timm.models import register_model

from grn.models.basic import FastRMSNorm, SelfAttnBlock
from grn.models.rope import precompute_rope3d_freqs_grid
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input


class MultipleLayers(nn.Module):
    """A sequential container for a chunk of multiple transformer blocks."""

    def __init__(self, layers: List[nn.Module], num_blocks: int, start_index: int):
        super().__init__()
        self.module = nn.ModuleList([
            layers[i] for i in range(start_index, start_index + num_blocks)
        ])

    def forward(
        self, x, cu_seqlens, max_seqlen, e0: Optional[torch.Tensor],
        attn_bias_or_two_vector: Optional[Any], attn_fn: Optional[Any] = None,
        checkpointing_full_block: bool = False, rope2d_freqs_grid: Optional[torch.Tensor] = None,
        scale_ind: Optional[Any] = None, context_info: Optional[Any] = None,
        last_diffusion_step: bool = True, ref_text_scale_inds: Optional[List[Any]] = None,
        use_cfg: bool = False, split_cond_uncond: Optional[List[Any]] = None
    ) -> torch.Tensor:
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(
                    m, h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector, attn_fn,
                    rope2d_freqs_grid, scale_ind, context_info,
                    last_diffusion_step, ref_text_scale_inds,
                    use_cfg, split_cond_uncond, use_reentrant=False
                )
            else:
                h = m(
                    h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector, attn_fn,
                    rope2d_freqs_grid, scale_ind, context_info,
                    last_diffusion_step, ref_text_scale_inds,
                    use_cfg, split_cond_uncond
                )
        return h


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal embeddings.

    Args:
        dim (int): Embedding dimension (must be even).
        position (torch.Tensor): Position tensor of shape [B, L].

    Returns:
        torch.Tensor: Embeddings of shape [B, L, dim].
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")

    half = dim // 2
    b, l = position.shape
    position = position.reshape(-1).type(torch.float64)

    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.reshape(b, l, dim)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def bld_to_bthwd(item: torch.Tensor, patch_time: int, patch_height: int, patch_width: int, apply_spatial_patchify: bool = False) -> torch.Tensor:
    """Reshape a sequence tensor to a spatial tensor."""
    batch_size = item.shape[0]
    return item.reshape(batch_size, patch_time, patch_height, patch_width, -1)


def build_attn_mask(seqlens, device):
    attn_mask = torch.zeros((1, 1, sum(seqlens), sum(seqlens)), dtype=torch.bool, device=device)
    q_start = 0
    for i in range(len(seqlens)):
        q_len = seqlens[i]
        q_end = q_start + q_len
        attn_mask[:, :, q_start:q_end, q_start:q_end] = True
        q_start = q_end
    return attn_mask


def update_finite_bit_lock(
    instant_candidate: torch.Tensor,
    remaining_steps: Optional[torch.Tensor],
    lock_steps: int,
):
    """Keep edit ownership alive for K updates without freezing bit values."""
    if lock_steps < 1:
        raise ValueError(f'lock_steps must be positive, got {lock_steps}')
    if instant_candidate.dtype != torch.bool:
        raise TypeError(
            f'instant_candidate must be bool, got {instant_candidate.dtype}'
        )
    if remaining_steps is None:
        remaining_steps = torch.zeros_like(
            instant_candidate,
            dtype=torch.int32,
        )
    elif remaining_steps.shape != instant_candidate.shape:
        raise ValueError(
            'remaining_steps and instant_candidate must have the same shape, '
            f'got {remaining_steps.shape} and {instant_candidate.shape}'
        )

    previously_active = remaining_steps > 0
    decayed_steps = torch.clamp(remaining_steps - 1, min=0)
    refreshed_steps = torch.full_like(decayed_steps, lock_steps)
    next_remaining_steps = torch.where(
        instant_candidate,
        refreshed_steps,
        decayed_steps,
    )
    active_mask = next_remaining_steps > 0
    carried_only_mask = active_mask & ~instant_candidate
    expired_mask = previously_active & ~active_mask
    return (
        next_remaining_steps,
        active_mask,
        carried_only_mask,
        expired_mask,
    )


class FsqHead(nn.Module):
    """Classification head for Finite Scalar Quantization (FSQ)."""

    def __init__(self, hidden_dim: int, fsq_dim: int, fsq_lvl: int, use_ada_layer_norm: bool, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, fsq_dim * fsq_lvl)
        self.norm = FastRMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return self.proj(self.norm(x))


class GRN(nn.Module):
    def __init__(
        self,
        vae_local: Any,
        arch: str = 'var',
        qwen_qkvo_bias: bool = False,
        text_channels: int = 0,
        text_maxlen: int = 0,
        embed_dim: int = 1024,
        depth: int = 16,
        num_key_value_heads: int = -1,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        block_chunks: int = 1,
        checkpointing: Optional[str] = None,
        pad_to_multiplier: int = 0,
        use_flex_attn: bool = False,
        num_of_label_value: int = 2,
        rope2d_normalized_by_hw: int = 0,
        pn: Optional[str] = None,
        video_frames: int = 1,
        always_training_scales: int = 20,
        apply_spatial_patchify: int = 0,
        inference_mode: bool = False,
        other_args: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.norm_eps = norm_eps
        self.drop_path_rate = drop_path_rate
        self.use_flex_attn = use_flex_attn
        self.checkpointing = checkpointing
        self.inference_mode = inference_mode
        self.other_args = other_args


        self.vae_embed_dim = vae_local.codebook_dim
        self.apply_spatial_patchify = apply_spatial_patchify
        self.text_channels = text_channels
        self.text_maxlen = text_maxlen
        self.is_text_to_image = text_channels != 0

        classifier_head_dim = other_args.detail_scale_dim
        classifier_head_lvl = other_args.detail_num_lvl
        hbq_round = other_args.hbq_round

        if other_args.refine_mode in ['ar_discrete_GRN_ind']:
            self.visual_embedding_in_dim = vae_local.codebook_dim * (2**hbq_round)
            classifier_head_dim = vae_local.codebook_dim
        elif other_args.refine_mode in ['ar_discrete_GRN_bit']:
            self.visual_embedding_in_dim = hbq_round * vae_local.codebook_dim * 2
            classifier_head_dim = hbq_round * vae_local.codebook_dim
        else:
            self.visual_embedding_in_dim = vae_local.codebook_dim

        if self.apply_spatial_patchify:
            self.visual_embedding_in_dim *= 4


        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.num_of_label_value = num_of_label_value
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(
            other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames
        )
        self.train_h_div_w_list = self.h_div_w_templates
        print(f"train_h_div_w_list: {self.train_h_div_w_list}")


        self.entrophy_statistics = []
        self.top_p, self.top_k = 1.0, 100
        self.maybe_record_function = nullcontext
        self.infer_ts = None


        self.norm0_cond = nn.Identity()
        self.text_proj = nn.Linear(self.text_channels, self.embed_dim)

        if self.other_args.use_ada_layer_norm:
            self.scale_or_time_dim = 256
            self.scale_or_time_embedding = nn.Sequential(
                nn.Linear(self.scale_or_time_dim, self.embed_dim), nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.scale_or_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim * 6))

        tmp_h_div_w_template = self.train_h_div_w_list[0]


        with torch.amp.autocast('cuda', dtype=torch.float32):
            self.rope2d_freqs_grid = precompute_rope3d_freqs_grid(
                dim=self.embed_dim // self.num_heads,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                activated_h_div_w_templates=self.train_h_div_w_list,
                max_scales=1010,
                max_frames=int(self.video_frames / other_args.temporal_compress_rate + 1),
                max_height=1800 // 8,
                max_width=1800 // 8,
                text_maxlen=self.text_maxlen,
                args=other_args,
            )

        self.word_embed = nn.Linear(self.visual_embedding_in_dim, self.embed_dim)
        self.head = FsqHead(
            hidden_dim=self.embed_dim,
            fsq_dim=classifier_head_dim,
            fsq_lvl=classifier_head_lvl,
            use_ada_layer_norm=other_args.use_ada_layer_norm,
        )

        if other_args.add_scale_token > 0:
            self.pt_embedder = TimestepEmbedder(self.embed_dim)


        self.attn_fn_compile_dict = {}
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                mlp_ratio=mlp_ratio,
                use_flex_attn=use_flex_attn,
                qwen_qkvo_bias=qwen_qkvo_bias,
                use_ada_layer_norm=other_args.use_ada_layer_norm,
            )
            self.unregistered_blocks.append(block)

        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // self.num_block_chunks
        assert self.num_blocks_in_a_chunk * self.num_block_chunks == depth, "Depth must be divisible by block_chunks"

        self.block_chunks = nn.ModuleList([
            MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i * self.num_blocks_in_a_chunk)
            for i in range(self.num_block_chunks)
        ])

        print(f"    [Model Config] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, "
              f"mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}")
        print(f"    drop_path_rate={drop_path_rate:g}", end='\n\n', flush=True)


    def get_logits_during_infer(self, hidden_states: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get logits during inference."""
        return self.head(hidden_states.float(), e)


    def prepare_text_conditions(
        self,
        label_B_or_BLT: Tuple[torch.Tensor, ...],
        negative_label_B_or_BLT: Optional[Tuple[torch.Tensor, ...]],
        use_cfg: bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Prepare text conditions for inference."""
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if use_cfg:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0)
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            lens = lens + lens_un
        kv_compact = self.text_proj(kv_compact).contiguous()
        return kv_compact, lens

    def embeds_codes2input(self, last_stage: torch.Tensor) -> torch.Tensor:
        """Embed discrete codes into continuous input representations."""
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
        last_stage = torch.permute(last_stage, [0, 2, 1])
        last_stage = self.word_embed(last_stage)
        return last_stage

    @torch.no_grad()
    def refine_edit(
        self,
        vae: Optional[Any] = None,
        scale_schedule: Optional[List[Tuple[int, int, int]]] = None,
        label_B_or_BLT: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        negative_label_B_or_BLT: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        g_seed: Optional[int] = None,
        cfg_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
        gt_leak: int = 0,
        args: Optional[Any] = None,
        get_visual_rope_embeds: Optional[Any] = None,
        noise_list: Optional[List[torch.Tensor]] = None,
        uncond_class_token_id: int = 1000,
        first_frame_condition: bool = False,
        **kwargs: Any,
    ):
        """Source refinement, AdaSF, FBL, and source-anchored bit routing."""
        if cfg_list is None: cfg_list = []
        if tau_list is None: tau_list = []

        from grn.schedules.global_refine import shift_pt


        assert len(cfg_list) >= len(scale_schedule), "Not enough CFG values for scales"
        assert len(tau_list) >= len(scale_schedule), "Not enough tau values for scales"

        ret, idx_Bl_list = [], []
        for b in self.unregistered_blocks: b.attn.kv_caching(True)

        total_steps = args.max_infer_steps
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks
        use_cfg = True
        cfg_interval = float(args.cfg_type.split('_')[-1])

        full_pt, ph, pw = scale_schedule[0]
        if first_frame_condition:
            pt = full_pt - 1
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (pt, ph, pw), 'cuda', args.mapped_h_div_w_template, t_offset=1)
        else:
            pt = full_pt

            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (pt, ph, pw), 'cuda', args.mapped_h_div_w_template, t_offset=0)


        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to('cuda')

        all_prefix_tokens = []
        all_lens = []
        for i in range(len(label_B_or_BLT)):
            prefix_tokens_i, lens_i = self.prepare_text_conditions(label_B_or_BLT[i], negative_label_B_or_BLT, use_cfg)
            all_prefix_tokens.append(prefix_tokens_i)
            all_lens.append(lens_i)


        device = all_prefix_tokens[0].device
        infer_device = all_prefix_tokens[0].device
        infer_dtype = all_prefix_tokens[0].dtype

        base_seed = 42 if g_seed is None else int(g_seed)
        rng = torch.Generator(device=device)
        rng.manual_seed(base_seed)
        edit_rng = torch.Generator(device=device)
        edit_rng.manual_seed(base_seed + 1)


        all_prefix_tokens_split = []
        for i in range(len(label_B_or_BLT)):
            split_i = torch.split(all_prefix_tokens[i], all_lens[i], dim=0)
            all_prefix_tokens_split.append(split_i)


        all_rope_cache_text_cond = []
        all_rope_cache_text_uncond = []
        for i in range(len(label_B_or_BLT)):
            rope_cache_text_cond_i = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:all_lens[i][0]]
            rope_cache_text_uncond_i = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:all_lens[i][1]]
            all_rope_cache_text_cond.append(rope_cache_text_cond_i)
            all_rope_cache_text_uncond.append(rope_cache_text_uncond_i)

        B = len(label_B_or_BLT)

        if args.refine_mode in ['ar_discrete_GRN_bit']:
            classes = 2

            labels_shape = (B,args.detail_scale_dim*args.hbq_round,pt,ph,pw)
        elif args.refine_mode in ['ar_discrete_GRN_index']:

            classes = 2**args.hbq_round

            labels_shape = (B,args.detail_scale_dim,pt,ph,pw)


        mul_pt_ph_pw = pt * ph * pw
        repeat_idx = -1

        scale_token_rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+args.add_scale_token]
        if noise_list is not None:
            absolute_gt_labels = noise_list[0].to('cuda').permute(0,2,3,4,1)
        assert len(scale_schedule) == 1
        if first_frame_condition:
            first_frame_labels = noise_list[0][:,:,:1]
            first_frame_tokens_cond = self.embeds_codes2input(multiclass_labels2onehot_input(first_frame_labels, classes))
            fist_frame_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (1, ph, pw), device, args.mapped_h_div_w_template, t_offset=0)
            visual_rope_cache = torch.cat((visual_rope_cache, fist_frame_rope_cache), dim=4)


        visual_token_count = visual_rope_cache.shape[4]
        tmp_seqlens = []
        rope_parts = []
        for i in range(B):
            tmp_seqlens.extend([
                visual_token_count + all_lens[i][0] + args.add_scale_token,
                visual_token_count + all_lens[i][1] + args.add_scale_token,
            ])
            rope_parts.extend([
                visual_rope_cache, all_rope_cache_text_cond[i], scale_token_rope_cache,
                visual_rope_cache, all_rope_cache_text_uncond[i], scale_token_rope_cache,
            ])
        rope_cache = torch.cat(rope_parts, dim=4)
        rope_cache = rope_cache[:,0].permute(0, 1, 3, 2, 4)

        cu_seqlens = torch.tensor([0]+tmp_seqlens, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(tmp_seqlens)

        source_tmp_seqlens = tmp_seqlens[:2]
        source_total_seqlen = sum(source_tmp_seqlens)
        source_rope_cache = rope_cache[:, :, :source_total_seqlen]
        source_cu_seqlens = cu_seqlens[:3]
        source_max_seqlen = max(source_tmp_seqlens)


        source_rand_labels = torch.randint(
            low=0,
            high=classes,
            size=(1, *labels_shape[1:]),
            device=infer_device,
            dtype=infer_dtype,
            generator=rng,
        )


        pure_rand_labels = source_rand_labels.repeat(B, 1, 1, 1, 1)
        mixed_xt = pure_rand_labels.clone()
        next_pt = 0.
        switch_step = getattr(args, 'switch_step', 12)
        tau_power = getattr(args, 'tau_power', 0.12)
        tau_spatial = getattr(args, 'tau_spatial', 0.015)
        adaptive_spatial_freezing = bool(
            getattr(args, 'adaptive_spatial_freezing', True)
        )
        freeze_multiplier = float(
            getattr(args, 'freeze_multiplier', 2.0)
        )
        configured_late_freeze_step = int(
            getattr(args, 'late_freeze_step', -1)
        )
        late_freeze_step = (
            args.max_infer_steps - 1
            if configured_late_freeze_step < 0
            else configured_late_freeze_step
        )
        bit_lock_steps = int(
            getattr(args, 'bit_lock_steps', 4)
        )
        mask_output_dir = getattr(args, 'mask_output_dir', None)
        mask_interval = int(getattr(args, 'mask_interval', 3))
        if bit_lock_steps <= 0:
            raise ValueError(
                f'bit_lock_steps must be positive, got '
                f'{bit_lock_steps}'
            )
        if mask_interval <= 0:
            raise ValueError(f'mask_interval must be positive, got {mask_interval}')
        if B > 2:
            raise ValueError(f'Source/edit inference expects at most two prompts, got {B}')
        if B == 2 and not 0 <= switch_step < args.max_infer_steps:
            raise ValueError(
                f'switch_step must be in [0, {args.max_infer_steps - 1}], '
                f'got {switch_step}'
            )
        if adaptive_spatial_freezing and not (
            switch_step <= late_freeze_step < args.max_infer_steps
        ):
            raise ValueError(
                f'late_freeze_step must be in '
                f'[{switch_step}, {args.max_infer_steps - 1}], '
                f'got {late_freeze_step}'
            )


        attn_mask = build_attn_mask(tmp_seqlens, device) if args.use_slow_attn else None
        source_attn_mask = (
            attn_mask[:, :, :source_total_seqlen, :source_total_seqlen]
            if attn_mask is not None else None
        )
        edit_initialized = False
        final_active_bit_mask_dthw = None
        frozen_spatial_mask = None
        adaptive_freeze_step = None
        initial_response = None
        bit_lock_remaining = None
        bit_lock_stats_initialized = False
        for cur_inner_round_si in range(args.max_infer_steps):
            cur_pt = next_pt
            is_last_step = np.abs(cur_pt - 1) < 0.02
            if cur_inner_round_si == 0:

                self.entrophy_statistics.append([])
            repeat_idx += 1

            cfg = cfg_list[0] if cur_pt >= cfg_interval else 1.0


            edit_active = (cur_inner_round_si >= switch_step and B == 2)
            if edit_active and not edit_initialized:


                mixed_xt[1] = mixed_xt[0].clone()
                edit_initialized = True
                print(f'[RefineEdit] initialize edit branch from source at step={cur_inner_round_si}')
                print(
                    f'[RefineEdit] dynamic spatial/bit split enabled: '
                    f'spatial_threshold={tau_spatial:.4f} '
                    f'bit_threshold={tau_power:.4f} '
                    f'bit_lock_steps={bit_lock_steps}'
                )


            active_prompt_count = B if edit_active else 1
            if edit_active:
                active_model_count = 2
                generation_model_indices = [0, 1]
                visual_state_indices = [0, 1]
                prompt_indices = [0, 1]
                active_tmp_seqlens = tmp_seqlens
                active_rope_cache = rope_cache
                active_cu_seqlens = cu_seqlens
                active_max_seqlen = max_seqlen
                active_attn_mask = attn_mask
            else:
                active_model_count = 1
                generation_model_indices = [0]
                visual_state_indices = [0]
                prompt_indices = [0]
                active_tmp_seqlens = source_tmp_seqlens
                active_rope_cache = source_rope_cache
                active_cu_seqlens = source_cu_seqlens
                active_max_seqlen = source_max_seqlen
                active_attn_mask = source_attn_mask


            active_mixed_xt = torch.cat([
                mixed_xt[state_i:state_i+1]
                for state_i in visual_state_indices
            ], dim=0)
            last_stage = self.embeds_codes2input(
                multiclass_labels2onehot_input(active_mixed_xt, classes)
            )
            pt_tokens = self.pt_embedder(
                torch.tensor([cur_pt], device=device)
            ).unsqueeze(0).repeat(active_model_count, 1, 1)

            all_last_stage_cond = []
            all_last_stage_uncond = []
            for i in range(active_model_count):
                visual_i = last_stage[i:i+1]
                if first_frame_condition:
                    state_i = visual_state_indices[i]
                    visual_i = torch.cat(
                        (visual_i, first_frame_tokens_cond[state_i:state_i+1]), dim=1
                    )
                prompt_i = prompt_indices[i]
                cond_i = torch.cat((
                    visual_i,
                    all_prefix_tokens_split[prompt_i][0].unsqueeze(0),
                    pt_tokens[i:i+1],
                ), dim=1)
                uncond_i = torch.cat((
                    visual_i,
                    all_prefix_tokens_split[prompt_i][1].unsqueeze(0),
                    pt_tokens[i:i+1],
                ), dim=1)
                all_last_stage_cond.append(cond_i)
                all_last_stage_uncond.append(uncond_i)

            parts = []
            for i in range(active_model_count):
                parts.append(all_last_stage_cond[i])
                parts.append(all_last_stage_uncond[i])
            last_stage = torch.cat(parts, dim=1)

            e, e0 = None, None
            last_diffusion_step = False


            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cu_seqlens=active_cu_seqlens, max_seqlen=active_max_seqlen, e0=e0, attn_bias_or_two_vector=active_attn_mask, attn_fn=None, rope2d_freqs_grid=active_rope_cache, last_diffusion_step=last_diffusion_step)

            logits = self.get_logits_during_infer(last_stage, e=e)
            tmp_bs, tmp_seq_len = logits.shape[:2]

            logits = logits.reshape(tmp_bs, tmp_seq_len, -1, args.detail_num_lvl)


            all_pred_cond_logits = []
            all_pred_uncond_logits = []
            offset = 0
            for i in range(active_model_count):
                cond_start = offset
                cond_end = offset + active_tmp_seqlens[2*i]
                uncond_start = cond_end
                uncond_end = cond_end + active_tmp_seqlens[2*i+1]

                all_pred_cond_logits.append(logits[:, cond_start:cond_start+mul_pt_ph_pw])
                all_pred_uncond_logits.append(logits[:, uncond_start:uncond_start+mul_pt_ph_pw])
                offset = uncond_end


            all_pred_cond_probs = [p.softmax(-1) for p in all_pred_cond_logits]


            categories = all_pred_cond_logits[0].shape[-1]


            entrophy = sum([
                (-all_pred_cond_probs[i] * torch.log2(all_pred_cond_probs[i]))
                .sum(-1).mean().item() / np.log2(categories)
                for i in generation_model_indices
            ]) / active_prompt_count


            pt_unshift = (cur_inner_round_si + 1) / (args.complexity_aware_Tmax - 1)
            pt_shift = shift_pt(min(1., pt_unshift), args.snr_shift)
            next_pt = 1 - np.cos(np.pi/2*pt_shift)
            next_pt = next_pt * 0.95


            all_pred_cond_labels = []
            all_pred_cfg_probs = []
            for model_i in generation_model_indices:
                pred_cond_labels_i = torch.argmax(all_pred_cond_probs[model_i], dim=-1)
                all_pred_cond_labels.append(bld_to_bthwd(pred_cond_labels_i, pt, ph, pw))
            for model_i in range(active_model_count):
                if cfg != 1:
                    pred_cfg_logits_i = all_pred_uncond_logits[model_i] + cfg * (all_pred_cond_logits[model_i] - all_pred_uncond_logits[model_i])
                else:
                    pred_cfg_logits_i = all_pred_cond_logits[model_i]
                pred_cfg_logits_i = pred_cfg_logits_i.mul(1/tau_list[0])
                all_pred_cfg_probs.append(pred_cfg_logits_i.softmax(dim=-1))


            all_pred_cfg_labels = []
            all_pred_sample_labels = []
            all_pred_sample_probs = []
            all_pred_sample_labels_thwd = []
            for branch_i, model_i in enumerate(generation_model_indices):
                pred_cfg_labels_i = torch.argmax(all_pred_cfg_probs[model_i], dim=-1)
                all_pred_cfg_labels.append(bld_to_bthwd(pred_cfg_labels_i, pt, ph, pw))

                branch_rng = rng if branch_i == 0 else edit_rng
                pred_sample_labels_i = torch.multinomial(
                    all_pred_cfg_probs[model_i].view(-1, args.detail_num_lvl),
                    num_samples=1,
                    replacement=True,
                    generator=branch_rng,
                ).view(1, mul_pt_ph_pw, -1)
                pred_sample_probs_i = torch.gather(all_pred_cfg_probs[model_i], dim=3, index=pred_sample_labels_i.unsqueeze(-1)).squeeze(-1)
                all_pred_sample_probs.append(bld_to_bthwd(pred_sample_probs_i, pt, ph, pw))
                all_pred_sample_labels.append(bld_to_bthwd(pred_sample_labels_i, pt, ph, pw))
                all_pred_sample_labels_thwd.append(pred_sample_labels_i)


            active_bit_mask = None
            if edit_active:
                source_sample_labels = all_pred_sample_labels_thwd[0]
                P_src = torch.gather(
                    all_pred_cfg_probs[0],
                    dim=3,
                    index=source_sample_labels.unsqueeze(-1),
                ).squeeze(-1)
                P_edit = torch.gather(
                    all_pred_cfg_probs[1],
                    dim=3,
                    index=source_sample_labels.unsqueeze(-1),
                ).squeeze(-1)
                probability_difference = P_src - P_edit

                spatial_score = probability_difference.mean(dim=-1)
                dynamic_spatial_mask = (
                    spatial_score > tau_spatial
                )
                spatial_mask_for_edit = dynamic_spatial_mask
                if adaptive_spatial_freezing:
                    if cur_inner_round_si == switch_step:
                        if dynamic_spatial_mask.any():
                            initial_response = (
                                spatial_score[dynamic_spatial_mask]
                                - tau_spatial
                            ).mean().item()
                        else:
                            initial_response = 0.0
                        strong_boundary = (
                            freeze_multiplier
                            * tau_spatial
                        )
                        is_early = initial_response >= strong_boundary
                        adaptive_freeze_step = (
                            switch_step
                            if is_early
                            else late_freeze_step
                        )
                        print(
                            f'[AdaSF] initial_response='
                            f'{initial_response:.6f} '
                            f'boundary={strong_boundary:.6f} '
                            f'regime={"early" if is_early else "late"} '
                            f'freeze_step={adaptive_freeze_step}'
                        )
                    if (
                        frozen_spatial_mask is None
                        and adaptive_freeze_step is not None
                        and cur_inner_round_si >= adaptive_freeze_step
                    ):
                        frozen_spatial_mask = dynamic_spatial_mask.clone()
                        print(
                            f'[AdaSF] spatial mask frozen at '
                            f'step={cur_inner_round_si} '
                            f'ratio={frozen_spatial_mask.float().mean().item():.3f}'
                        )
                    if frozen_spatial_mask is not None:
                        spatial_mask_for_edit = frozen_spatial_mask
                raw_bit_mask = (
                    probability_difference > tau_power
                )
                instant_candidate_bit_mask = (
                    raw_bit_mask & spatial_mask_for_edit.unsqueeze(-1)
                )
                (
                    bit_lock_remaining,
                    active_bit_mask,
                    carried_only_bit_mask,
                    expired_bit_mask,
                ) = update_finite_bit_lock(
                    instant_candidate=instant_candidate_bit_mask,
                    remaining_steps=bit_lock_remaining,
                    lock_steps=bit_lock_steps,
                )

                active_bit_mask_thwd = bld_to_bthwd(
                    active_bit_mask, pt, ph, pw
                )
                active_bit_mask_dthw = active_bit_mask_thwd.permute(
                    0, 4, 1, 2, 3
                ).contiguous()

                dynamic_spatial_ratio = (
                    dynamic_spatial_mask.float().mean().item()
                )
                applied_spatial_ratio = (
                    spatial_mask_for_edit.float().mean().item()
                )
                raw_bit_ratio = raw_bit_mask.float().mean().item()
                instant_candidate_bit_ratio = (
                    instant_candidate_bit_mask.float().mean().item()
                )
                carried_only_bit_ratio = (
                    carried_only_bit_mask.float().mean().item()
                )
                expired_bit_ratio = expired_bit_mask.float().mean().item()
                active_bit_ratio = active_bit_mask.float().mean().item()
                effective_spatial_ratio = (
                    active_bit_mask.any(dim=-1).float().mean().item()
                )
                print(
                    f'[RefineEdit] step={cur_inner_round_si} '
                    f'spatial_driver='
                    f'{"source_init" if cur_inner_round_si == switch_step else "edit_trajectory"} '
                    f'dynamic_spatial_ratio={dynamic_spatial_ratio:.3f} '
                    f'applied_spatial_ratio={applied_spatial_ratio:.3f} '
                    f'spatial_frozen={int(frozen_spatial_mask is not None)} '
                    f'raw_bit_ratio={raw_bit_ratio:.3f} '
                    f'instant_candidate_bit_ratio='
                    f'{instant_candidate_bit_ratio:.3f} '
                    f'carried_only_bit_ratio={carried_only_bit_ratio:.3f} '
                    f'expired_bit_ratio={expired_bit_ratio:.3f} '
                    f'active_bit_ratio={active_bit_ratio:.3f} '
                    f'effective_spatial_ratio={effective_spatial_ratio:.3f}'
                )

                if mask_output_dir is not None:
                    os.makedirs(mask_output_dir, exist_ok=True)
                    stats_path = os.path.join(
                        mask_output_dir,
                        'adaptive_bit_lock_stats.csv',
                    )
                    stats_mode = 'a' if bit_lock_stats_initialized else 'w'
                    with open(stats_path, stats_mode, encoding='utf-8') as f:
                        if not bit_lock_stats_initialized:
                            f.write(
                                'step,cur_pt,lock_steps,initial_response,'
                                'freeze_step,spatial_frozen,'
                                'dynamic_spatial_ratio,applied_spatial_ratio,'
                                'raw_bit_ratio,instant_candidate_bit_ratio,'
                                'carried_only_bit_ratio,expired_bit_ratio,'
                                'active_bit_ratio,effective_spatial_ratio\n'
                            )
                        response_value = (
                            '' if initial_response is None
                            else f'{initial_response:.8f}'
                        )
                        freeze_value = (
                            '' if adaptive_freeze_step is None
                            else str(adaptive_freeze_step)
                        )
                        f.write(
                            f'{cur_inner_round_si},{cur_pt:.8f},'
                            f'{bit_lock_steps},{response_value},'
                            f'{freeze_value},'
                            f'{int(frozen_spatial_mask is not None)},'
                            f'{dynamic_spatial_ratio:.8f},'
                            f'{applied_spatial_ratio:.8f},'
                            f'{raw_bit_ratio:.8f},'
                            f'{instant_candidate_bit_ratio:.8f},'
                            f'{carried_only_bit_ratio:.8f},'
                            f'{expired_bit_ratio:.8f},'
                            f'{active_bit_ratio:.8f},'
                            f'{effective_spatial_ratio:.8f}\n'
                        )
                    bit_lock_stats_initialized = True

                if (
                    mask_output_dir is not None
                    and (cur_inner_round_si - switch_step)
                    % mask_interval == 0
                ):
                    mask_image_size = tuple(
                        getattr(args, 'mask_image_size', (ph, pw))
                    )


                    panel_items = []
                    spatial_mask_hw = spatial_mask_for_edit.reshape(
                        1, pt, ph, pw
                    )[0, 0]
                    spatial_display = (~spatial_mask_hw).to(
                        torch.float32
                    )[None, None]
                    spatial_display = F.interpolate(
                        spatial_display,
                        size=mask_image_size,
                        mode='nearest',
                    )[0, 0]
                    spatial_array = (
                        spatial_display.mul(255).to(torch.uint8).cpu().numpy()
                    )
                    panel_items.append(
                        ('applied spatial mask', Image.fromarray(spatial_array))
                    )

                    active_bit_density_hw = active_bit_mask_thwd[0, 0].to(
                        torch.float32
                    ).mean(dim=-1)
                    active_bit_display = (
                        1.0 - active_bit_density_hw
                    )[None, None]
                    active_bit_display = F.interpolate(
                        active_bit_display,
                        size=mask_image_size,
                        mode='nearest',
                    )[0, 0]
                    active_bit_array = (
                        active_bit_display.mul(255).round().to(torch.uint8)
                        .cpu().numpy()
                    )
                    panel_items.append(
                        (
                            'active bit density',
                            Image.fromarray(active_bit_array),
                        )
                    )

                    tile_image_height, tile_width = mask_image_size
                    label_height = 24
                    tile_height = tile_image_height + label_height
                    summary_panel = Image.new(
                        'L',
                        (tile_width * 2, tile_height),
                        color=255,
                    )
                    panel_draw = ImageDraw.Draw(summary_panel)
                    for panel_index, (mask_label, mask_image) in enumerate(
                        panel_items
                    ):
                        panel_x = (panel_index % 2) * tile_width
                        panel_y = 0
                        panel_draw.text(
                            (panel_x + 4, panel_y + 5),
                            mask_label,
                            fill=0,
                        )
                        summary_panel.paste(
                            mask_image,
                            (panel_x, panel_y + label_height),
                        )

                    mask_dir = os.path.join(
                        mask_output_dir, 'mask_summary'
                    )
                    os.makedirs(mask_dir, exist_ok=True)
                    mask_path = os.path.join(
                        mask_dir, f'step_{cur_inner_round_si:03d}.png'
                    )
                    summary_panel.save(mask_path)
                    print(f'[RefineEdit] saved mask summary: {mask_path}')


            pred_cond_labels = torch.cat(all_pred_cond_labels, dim=0)
            pred_cfg_labels = torch.cat(all_pred_cfg_labels, dim=0)
            pred_sample_labels = torch.cat(all_pred_sample_labels, dim=0)
            pred_sample_probs = torch.cat(all_pred_sample_probs, dim=0)


            assume_flip_ratio = (1 - cur_pt) / args.detail_num_lvl * 100.
            pred_zero_ratio = (pred_cond_labels == 0).sum() / pred_cond_labels.numel() * 100.
            pred_one_ratio = (pred_cond_labels == 1).sum() / pred_cond_labels.numel() * 100.
            mixed_xt_Bthwd_01 = mixed_xt[:active_prompt_count].clone().permute(0,2,3,4,1)
            mixed_xt_Bthwd_01[mixed_xt_Bthwd_01<0] = 0
            pred_cond_flip_ratio = (pred_cond_labels != mixed_xt_Bthwd_01).sum() / pred_cond_labels.numel() * 100.
            pred_cfg_flip_ratio = (pred_cfg_labels != mixed_xt_Bthwd_01).sum() / pred_cfg_labels.numel() * 100.
            pred_sample_flip_ratio = (pred_sample_labels != mixed_xt_Bthwd_01).sum() / pred_sample_labels.numel() * 100.
            self.entrophy_statistics[-1].append({
                'cur_inner_round_si': cur_inner_round_si,
                'cur_pt': cur_pt,


                'entrophy': entrophy,
                'assume_flip_ratio': assume_flip_ratio,
                'pred_cond_flip_ratio': pred_cond_flip_ratio.item(),
                'pred_cfg_flip_ratio': pred_cfg_flip_ratio.item(),
                'pred_sample_flip_ratio': pred_sample_flip_ratio.item(),
                'pred_zero_ratio': pred_zero_ratio.item(),
                'pred_one_ratio': pred_one_ratio.item(),
                'meta': args.meta,
            })
            print(f'{repeat_idx=} {cur_inner_round_si=} {cur_pt=:.3f} {pred_sample_labels.shape=}')
            print(f'{assume_flip_ratio=:.2f}% {pred_cond_flip_ratio=:.2f}% {pred_cfg_flip_ratio=:.2f}% {pred_sample_flip_ratio=:.2f}%')
            if repeat_idx < gt_leak:
                gt_labels = absolute_gt_labels[:active_prompt_count]
                gt_flip_ratio = (gt_labels != mixed_xt_Bthwd_01).sum() / gt_labels.numel() * 100.
                gt_flip_ratio = gt_flip_ratio.item()
                pred_cond_acc = (gt_labels==pred_cond_labels).to(float).mean().item()
                pred_cfg_acc = (gt_labels==pred_cfg_labels).to(float).mean().item()
                pred_sample_acc = (gt_labels==pred_sample_labels).to(float).mean().item()
                print(f'{repeat_idx=} {entrophy=:.4f} {pred_cond_acc=:.4f} {pred_cfg_acc=:.4f} {pred_sample_acc=:.4f}')
                self.entrophy_statistics[-1][-1].update({
                    'gt_flip_ratio': gt_flip_ratio,
                    'pred_cond_acc': pred_cond_acc,
                    'pred_cfg_acc': pred_cfg_acc,
                    'pred_sample_acc': pred_sample_acc,
                })
                pred_sample_labels = gt_labels

            pred_sample_labels = pred_sample_labels.permute(0,4,1,2,3)
            pred_sample_probs = pred_sample_probs.permute(0,4,1,2,3)

            use_predict_mask = torch.cat([
                torch.rand(
                    pred_sample_labels[i:i+1].shape,
                    device=device,
                    generator=rng if i == 0 else edit_rng,
                ) < next_pt
                for i in range(active_prompt_count)
            ], dim=0)

            mixed_xt[:active_prompt_count] = torch.where(
                use_predict_mask,
                pred_sample_labels,
                pure_rand_labels[:active_prompt_count],
            )


            if edit_active:
                active_bit_mask_step = active_bit_mask_dthw[0]
                mixed_xt[1] = torch.where(
                    active_bit_mask_step,
                    mixed_xt[1],
                    mixed_xt[0],
                )
                final_active_bit_mask_dthw = active_bit_mask_step


            next_pt = use_predict_mask[:1].float().mean().item()

            pbar.update(1)
            if is_last_step: break


        if final_active_bit_mask_dthw is not None:
            pred_sample_labels[1] = torch.where(
                final_active_bit_mask_dthw,
                pred_sample_labels[1],
                pred_sample_labels[0],
            )

        if first_frame_condition:
            pred_sample_labels = torch.cat((first_frame_labels, pred_sample_labels), dim=2)

        if args.refine_mode == 'ar_discrete_GRN_ind':
            from grn.utils_t2iv.hbq_util_t2iv import index_label2quant_features
            approx_signal = index_label2quant_features(pred_sample_labels, hbq_round=args.hbq_round)
        elif args.refine_mode == 'ar_discrete_GRN_bit':
            from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature
            approx_signal = bit_label2raw_feature(pred_sample_labels, hbq_round=args.hbq_round)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        img = self.summed_codes2images(vae, approx_signal)
        return ret, idx_Bl_list, img

    def summed_codes2images(self, vae: Any, summed_codes: torch.Tensor) -> torch.Tensor:
        """Decode summed codes into images using the VAE."""
        t1 = time.time()
        img = vae.decode(summed_codes, slice=True)
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(0, 2, 3, 4, 1)
        img = img.mul_(255).to(torch.uint8).flip(dims=(4,))
        print(f"Decode takes {time.time() - t1:.1f}s")
        return img


    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = False, assign: bool = False) -> Any:
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


    def extra_repr(self) -> str:
        return f'drop_path_rate={self.drop_path_rate}'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}


@register_model
def GRN2b(depth: int = 28, block_chunks: int = 7, embed_dim: int = 2304, num_heads: int = 18, num_key_value_heads: int = 18, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN:
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )
