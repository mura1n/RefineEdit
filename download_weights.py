"""Download only the three pretrained components required by RefineEdit."""

import argparse
from pathlib import Path

from refineedit.config import MODEL_FILENAME, VAE_FILENAME, checkpoint_paths, check_weights

OFFICIAL_REPOSITORY = "bytedance-research/GRN"
DEFAULT_REVISION = "3d4699d4d31fe5e0bf7cc8f25c4d98b315a87a18"


def main():
    parser = argparse.ArgumentParser(description="Download pretrained GRN T2I weights")
    parser.add_argument("--output-dir", type=Path, default=Path("weights"))
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Hugging Face commit, tag, or branch (defaults to a pinned revision)")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    # Resolve main once so all components are from the same revision.
    revision = HfApi().model_info(OFFICIAL_REPOSITORY, revision=args.revision).sha
    print(f"Downloading official GRN T2I components at revision {revision}")
    snapshot_download(
        repo_id=OFFICIAL_REPOSITORY, revision=revision, local_dir=str(args.output_dir),
        allow_patterns=[MODEL_FILENAME, VAE_FILENAME, "umt5-xxl/**"],
    )
    check_weights(checkpoint_paths(args.output_dir))
    (args.output_dir / "REVISION.txt").write_text(revision + "\n", encoding="utf-8")
    print("Weights are ready. No RefineEdit-specific checkpoint is required.")


if __name__ == "__main__":
    main()
