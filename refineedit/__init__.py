"""RefineEdit: training-free prompt-to-prompt image editing."""

from .config import RefineEditConfig

__all__ = ["RefineEditConfig", "RefineEditPipeline"]


def __getattr__(name):
    if name == "RefineEditPipeline":
        from .pipeline import RefineEditPipeline
        return RefineEditPipeline
    raise AttributeError(name)
