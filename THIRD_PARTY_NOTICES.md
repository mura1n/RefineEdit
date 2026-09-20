# Third-party components

The `grn/` modules are adapted from the public GRN implementation. Its MIT
license and copyright notice are retained in `LICENSE`. These are upstream
attributions, not information about the authors of RefineEdit.

GRN's pretrained T2I model, HBQ tokenizer, and UMT5 text encoder are downloaded
separately from the official `bytedance-research/GRN` model repository. No model
weights are distributed in this repository. Follow the terms accompanying those
models and all installed third-party dependencies.

Training entry points, distributed training utilities, video inference entry
points, dataset tooling, and experiment infrastructure are not included. Some
shared architecture names and image/video tensor layouts remain necessary for
compatibility with the published HBQ and GRN checkpoints.
