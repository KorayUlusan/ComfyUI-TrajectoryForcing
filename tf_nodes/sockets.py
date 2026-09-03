"""This extension's own socket types.

`Custom` registers a link colour and type-checked socket without ComfyUI needing
to know what travels on it, which is what the three payloads in `data.py` need:
a trajectory, a region map and a token selection are none of them images,
latents or masks in ComfyUI's sense.
"""
from __future__ import annotations

from comfy_api.latest import io

CATEGORY = "TrajectoryForcing"

TFPipelineSocket = io.Custom("TF_PIPELINE")
TFLevelsSocket = io.Custom("TF_LEVELS")
TFRegionsSocket = io.Custom("TF_REGIONS")
TFTokensSocket = io.Custom("TF_TOKENS")

# Every node that names a level does it with the same widget, so a level index
# means the same thing (and clamps the same way) across the whole graph.
MAX_LEVELS = 16


def level_input(id: str = "level", default: int = 2, tooltip: str = "") -> io.Int.Input:
    return io.Int.Input(
        id,
        default=default,
        min=0,
        max=MAX_LEVELS - 1,
        tooltip=tooltip or "Hierarchy level, 0 = coarsest (object/background). "
                          "Clamped to the model's level count.",
    )


def pipeline_input(tooltip: str = "") -> io.Custom("TF_PIPELINE").Input:
    """The pipeline socket, optional because a trajectory usually carries its own.

    TF Generate attaches the pipeline that produced it to the TF_LEVELS it
    outputs, so consumers can find it without a wire. The socket stays for the
    one case that has none -- a trajectory restored by TF Load Levels -- and as
    an override when two pipelines are in play.
    """
    return TFPipelineSocket.Input(
        "pipeline", optional=True,
        tooltip=tooltip or "Usually unnecessary: the trajectory carries the pipeline that "
                           "made it. Wire it for a trajectory from TF Load Levels.",
    )


def resolve_pipeline(pipeline, levels, what: str = "this node"):
    """The explicit pipeline if wired, else the one the trajectory carries."""
    found = pipeline if pipeline is not None else getattr(levels, "pipeline", None)
    if found is None:
        raise ValueError(
            f"{what} needs a pipeline, and this trajectory carries none -- it was most "
            "likely restored by TF Load Levels. Wire TF Load Pipeline into its "
            "'pipeline' input."
        )
    return found


def seed_input(default: int = 592) -> io.Int.Input:
    return io.Int.Input(
        "seed",
        default=default,
        min=0,
        max=0xFFFFFFFFFFFFFFFF,
        control_after_generate=True,
        tooltip="Sampling seed. Re-sampling with the same seed and class reproduces a trajectory exactly.",
    )
