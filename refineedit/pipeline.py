"""Single-device image inference using the pretrained GRN T2I backbone."""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image

from grn.models.grn import GRN2b
from grn.models.hbq_tokenizer import HBQ_Tokenizer
from grn.models.umt5.t5 import T5EncoderModel
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.schedules.global_refine import get_visual_rope_embeds
from .config import RefineEditConfig, TEXT_FILENAME, checkpoint_paths, check_weights


@dataclass
class RefineEditOutput:
    source: Image.Image
    editing: Image.Image

    @property
    def images(self):
        return [self.source, self.editing]


def _task_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompts must be nonempty strings")
    prompt = prompt.strip()
    if prompt == "<T2I>":
        raise ValueError("A prompt must contain text after <T2I>")
    return prompt if prompt.startswith("<T2I>") else "<T2I>" + prompt


class RefineEditPipeline:
    """Generate a source image and its edit from a pair of text prompts.

    This API does not accept arbitrary input images. It constructs the source
    refinement trajectory needed by bit routing. Calls must not run concurrently
    on the same pipeline instance.
    """

    @classmethod
    def from_pretrained(cls, weights_dir="weights", *, model_path=None,
                        vae_path=None, text_encoder_path=None, device="cuda",
                        attention_backend="flash"):
        if attention_backend not in {"flash", "sdpa"}:
            raise ValueError("attention_backend must be flash or sdpa")
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("RefineEdit inference requires a CUDA GPU with BF16 support")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
        if attention_backend == "flash":
            try:
                from flash_attn import flash_attn_varlen_func  # noqa: F401
            except ImportError as error:
                raise RuntimeError(
                    "Install FlashAttention 2 (see README), or use --attention-backend sdpa"
                ) from error

        paths = checkpoint_paths(weights_dir, model_path, vae_path, text_encoder_path)
        check_weights(paths)
        model_path, vae_path, text_path = paths
        args = SimpleNamespace(
            model="GRN2b", pn="1M", video_frames=81,
            vae_latent_dim=64, hbq_round=4, detail_scale_dim=64,
            detail_num_lvl=2, num_of_label_value=2, text_channels=4096,
            rope2d_normalized_by_hw=2, apply_spatial_patchify=0,
            dynamic_scale_schedule="GRN_vae_stride16", train_h_div_w_list="[]",
            temporal_compress_rate=4, use_ada_layer_norm=0, add_scale_token=1,
            refine_mode="ar_discrete_GRN_bit", cfg_type="cfg_interval_0.0",
            meta="", use_slow_attn=attention_backend == "sdpa",
            other_device=device,
        )
        # CPU construction matches the original T2I loading path and avoids
        # keeping a second checkpoint copy on the GPU during initialization.
        print("Loading UMT5 text encoder...")
        text_encoder = T5EncoderModel(
            text_len=512, dtype=torch.bfloat16, device="cpu",
            checkpoint_path=str(text_path / TEXT_FILENAME),
            tokenizer_path=str(text_path / "umt5-xxl"), enable_fsdp=False,
        )
        print("Loading HBQ tokenizer...")
        with torch.no_grad():
            vae = HBQ_Tokenizer(args=args, latent_channels=64,
                                encoder_out_type="feature_tanh").eval().requires_grad_(False)
            state = torch.load(vae_path, map_location="cpu", weights_only=True)
            vae.load_state_dict(state["ema"] if "ema" in state else state["vae"],
                                strict=True, assign=True)
            del state
            print("Loading GRN T2I backbone...")
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model = GRN2b(
                vae_local=vae, text_channels=4096, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None, checkpointing="full-block",
                customized_flash_attn=False, fused_norm=True, pad_to_multiplier=128,
                use_flex_attn=False, num_of_label_value=2,
                rope2d_normalized_by_hw=2, pn="1M", apply_spatial_patchify=0,
                inference_mode=True, train_h_div_w_list="[]",
                dynamic_scale_schedule=args.dynamic_scale_schedule,
                video_frames=args.video_frames, other_args=args,
            ).eval().requires_grad_(False)
            model.load_state_dict(state["trainer"]["gpt_fsdp"] if "trainer" in state else state,
                                  strict=True)
            del state
        pipeline = cls()
        pipeline.args = args
        pipeline.model = model.to(device)
        pipeline.vae = vae.to(device)
        pipeline.text_encoder = text_encoder
        pipeline.device = device
        pipeline.attention_backend = attention_backend
        return pipeline

    def _encode_prompt(self, prompt):
        self.text_encoder.model.to(self.device)
        features = self.text_encoder([prompt], self.device)
        lengths = [len(feature) for feature in features]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return (torch.cat(features, dim=0).float().to(self.device), lengths,
                torch.tensor(offsets, dtype=torch.int32), max(lengths))

    @torch.no_grad()
    def __call__(self, source_prompt, editing_prompt, *, config=None,
                 negative_prompt="", mask_output_dir=None, **parameters):
        if config is not None and parameters:
            raise ValueError("Pass either config or individual parameters, not both")
        config = config or RefineEditConfig(**parameters)
        if not isinstance(config, RefineEditConfig):
            raise TypeError("config must be a RefineEditConfig")
        prompts = [_task_prompt(source_prompt), _task_prompt(editing_prompt)]
        if not isinstance(negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        args = self.args
        for key, value in asdict(config).items():
            setattr(args, key, value)
        args.max_infer_steps = config.steps
        args.complexity_aware_Tmax = config.steps
        args.snr_shift = 1.0
        args.late_freeze_step = config.steps - 1
        args.mask_output_dir = str(mask_output_dir) if mask_output_dir is not None else None
        if mask_output_dir is not None:
            Path(mask_output_dir).mkdir(parents=True, exist_ok=True)
        resolutions, _ = get_dynamic_resolution_meta(
            args.dynamic_scale_schedule, args.train_h_div_w_list, args.video_frames)
        args.mapped_h_div_w_template = 1.0
        metadata = resolutions[1.0][args.pn]
        args.mask_image_size = metadata["pixel"]
        args.first_full_spatial_size_scale_index = 0
        args.tower_split_index = 1
        with torch.cuda.device(self.device):
            text = [self._encode_prompt(prompt) for prompt in prompts]
            negative = self._encode_prompt(negative_prompt)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
                _, _, images = self.model.refine_edit(
                    vae=self.vae, scale_schedule=metadata["pt2scale_schedule"][1],
                    label_B_or_BLT=text, negative_label_B_or_BLT=negative,
                    g_seed=config.seed, cfg_list=[config.guidance_scale],
                    tau_list=[config.temperature], gt_leak=-1, args=args,
                    get_visual_rope_embeds=get_visual_rope_embeds,
                    noise_list=None, first_frame_condition=False,
                )
        arrays = images.cpu().numpy()
        return RefineEditOutput(*(Image.fromarray(arrays[i, 0, ..., ::-1]) for i in range(2)))
