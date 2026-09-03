"""This extension's own socket types.

`Custom` registers a link colour and type-checked socket without ComfyUI needing
to know what travels on it, which is what the three payloads in `data.py` need:
a trajectory, a region map and a token selection are none of them images,
latents or masks in ComfyUI's sense.
"""
from __future__ import annotations

from comfy_api.latest import io, ui

# Subcategories mirror the README's own three sections. Eighteen nodes in one
# flat list is a scroll; the split is how you find "the node that picks a
# region" without already knowing its name. Search still spans all of them.
CATEGORY = "TrajectoryForcing"
CATEGORY_GENERATE = "TrajectoryForcing/generate"
CATEGORY_SELECT = "TrajectoryForcing/select"
CATEGORY_EDIT = "TrajectoryForcing/edit"
CATEGORY_IO = "TrajectoryForcing/save and load"


def node_preview(image=None, text: str = "") -> dict | None:
    """What a node shows in its own body.

    Stock ComfyUI ships no node that displays a STRING -- checked against every
    registered class -- so an `info` output is invisible unless the node that
    computed it shows it itself. Before this, forty-two text and number outputs
    across the four example workflows went nowhere: the edit summary, the resume
    summary, the region count, and TF Levels Info's entire reason to exist.

    Returns the merged dict rather than a `_UIOutput`, because a node that
    produces both a picture and a number should show both, and the two preview
    classes each own only their own key.
    """
    parts: dict = {}
    if image is not None:
        parts.update(ui.PreviewImage(image).as_dict())
    if text:
        parts.update(ui.PreviewText(text).as_dict())
    return parts or None

TFPipelineSocket = io.Custom("TF_PIPELINE")
TFLevelsSocket = io.Custom("TF_LEVELS")
TFRegionsSocket = io.Custom("TF_REGIONS")
TFTokensSocket = io.Custom("TF_TOKENS")

# Every node that names a level does it with the same widget, so a level index
# means the same thing (and clamps the same way) across the whole graph.
MAX_LEVELS = 16

# How many levels a released model has. `dataset.num_levels = 4` in configs/
# default.py and in every shipped config, because the hierarchy the method is
# built on has four rungs (object/background, parts, subparts, fine) -- it is a
# property of TF, not a per-run setting. Dropdowns that name levels one by one
# list this many, and `level_override` covers anything deeper.
SHIPPED_LEVELS = 4


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


# Every widget in this extension that can decide for itself uses -1 to mean so.
# One convention, stated on the widget rather than left in a tooltip nobody
# hovers before turning a knob.
AUTO = -1
AUTO_SUFFIX = "(-1 = auto)"


def auto_label(id: str) -> str:
    return f"{id.replace('_', ' ')} {AUTO_SUFFIX}"


def auto_level_input(id: str, means: str, tooltip: str) -> io.Int.Input:
    """A level widget where -1 means "work it out", so there is no dead knob.

    Two widgets where one silently disables the other is the shape that confuses
    people -- `which` + `level`, `follow_edit` + `level`. Folding the automatic
    case into the value itself leaves one widget that always does something.

    The label says `-1 = auto` outright: the sentinel is only obvious once you
    know the convention, and a widget reading plain `level` with a `-1` in it
    tells a first-time reader nothing.
    """
    return io.Int.Input(
        id, display_name=auto_label(id),
        default=AUTO, min=AUTO, max=MAX_LEVELS - 1, advanced=True,
        tooltip=f"-1 (auto) {means}. {tooltip}",
    )


def seed_input(default: int = 592) -> io.Int.Input:
    return io.Int.Input(
        "seed",
        default=default,
        min=0,
        max=0xFFFFFFFFFFFFFFFF,
        control_after_generate=True,
        tooltip="Sampling seed. Re-sampling with the same seed and class reproduces a trajectory exactly.",
    )
