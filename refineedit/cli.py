"""Command line interface with lightweight argument and batch validation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import RefineEditConfig, checkpoint_paths, check_weights


def read_pairs(path):
    pairs = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                item = [item.get("source_prompt"), item.get("editing_prompt")]
            if not isinstance(item, list) or len(item) != 2 or not all(
                isinstance(text, str) and text.strip() and text.strip() != "<T2I>"
                for text in item
            ):
                raise ValueError(f"Invalid prompt pair on line {line_number}")
            pairs.append(item)
    if not pairs:
        raise ValueError("The prompt file contains no pairs")
    return pairs


def parser():
    p = argparse.ArgumentParser(description="RefineEdit prompt-to-prompt image inference")
    p.add_argument("--source-prompt")
    p.add_argument("--editing-prompt")
    p.add_argument("--prompts-file", type=Path,
                   help="JSONL: one [source, editing] pair or object per line")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, help="Exclusive end index")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--model-path")
    p.add_argument("--vae-path")
    p.add_argument("--text-encoder-path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attention-backend", choices=("flash", "sdpa"), default="flash")
    p.add_argument("--switch-step", type=int, default=12)
    p.add_argument("--tau-spatial", type=float, default=0.015)
    p.add_argument("--tau-power", type=float, default=0.12)
    p.add_argument("--bit-lock-steps", type=int, default=4)
    p.add_argument("--freeze-multiplier", type=float, default=2.0)
    p.add_argument("--disable-adaptive-spatial-freezing", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--mask-interval", type=int, default=3)
    p.add_argument("--editing-only", action="store_true", help="Skip source-image saving")
    p.add_argument("--image-format", choices=("png", "jpg"), default="png")
    p.add_argument("--dry-run", action="store_true", help="Validate prompts/config without loading weights")
    return p


def main(argv=None):
    p = parser()
    options = p.parse_args(argv)
    if options.prompts_file is not None:
        if options.source_prompt is not None or options.editing_prompt is not None:
            p.error("Use --prompts-file OR --source-prompt and --editing-prompt")
        pairs = read_pairs(options.prompts_file)
    else:
        if not options.source_prompt or not options.editing_prompt:
            p.error("Provide both --source-prompt and --editing-prompt")
        pairs = [[options.source_prompt, options.editing_prompt]]
    if any(not text.strip() or text.strip() == "<T2I>" for pair in pairs for text in pair):
        p.error("Prompts must contain text")
    end = len(pairs) if options.end_index is None else options.end_index
    if not 0 <= options.start_index < end <= len(pairs):
        p.error("Invalid [start-index, end-index) sample range")
    config = RefineEditConfig(
        switch_step=options.switch_step, tau_spatial=options.tau_spatial,
        tau_power=options.tau_power, bit_lock_steps=options.bit_lock_steps,
        adaptive_spatial_freezing=not options.disable_adaptive_spatial_freezing,
        freeze_multiplier=options.freeze_multiplier, steps=options.steps,
        guidance_scale=options.guidance_scale, temperature=options.temperature,
        seed=options.seed, mask_interval=options.mask_interval,
    )
    print(json.dumps({"parameters": asdict(config), "resolution": [1024, 1024],
                      "sample_range": [options.start_index, end],
                      "attention_backend": options.attention_backend}, indent=2))
    if options.dry_run:
        return
    check_weights(checkpoint_paths(options.weights_dir, options.model_path,
                                   options.vae_path, options.text_encoder_path))
    # Refuse accidental overwrites of earlier inference runs.
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if any(options.output_dir.iterdir()):
        raise FileExistsError("output-dir must be empty; choose a new directory for this run")
    from .pipeline import RefineEditPipeline
    pipeline = RefineEditPipeline.from_pretrained(
        options.weights_dir, model_path=options.model_path, vae_path=options.vae_path,
        text_encoder_path=options.text_encoder_path, device=options.device,
        attention_backend=options.attention_backend)
    editing_dir = options.output_dir / "editing"
    editing_dir.mkdir()
    if not options.editing_only:
        (options.output_dir / "original").mkdir()
    for index in range(options.start_index, end):
        stem = f"{index:04d}"
        source, editing = pairs[index]
        print(f"RefineEdit sample {index + 1}/{end}")
        masks = options.output_dir / "masks" / stem if options.save_masks else None
        result = pipeline(source, editing, config=config,
                          negative_prompt=options.negative_prompt, mask_output_dir=masks)
        result.editing.save(editing_dir / f"{stem}.{options.image_format}")
        if not options.editing_only:
            result.source.save(options.output_dir / "original" / f"{stem}.{options.image_format}")
        # Store reproducible settings, but not local checkpoint paths or host metadata.
        record = {"index": index, "source_prompt": source, "editing_prompt": editing,
                  "negative_prompt": options.negative_prompt, "parameters": asdict(config),
                  "attention_backend": options.attention_backend,
                  "resolution": [1024, 1024], "image_format": options.image_format}
        with (options.output_dir / "inference.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Completed {end - options.start_index} prompt pair(s).")
