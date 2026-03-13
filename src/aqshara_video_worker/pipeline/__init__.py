from __future__ import annotations

__all__ = ["PipelineRunner"]


def __getattr__(name: str):
    if name == "PipelineRunner":
        from .runner import PipelineRunner

        return PipelineRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
