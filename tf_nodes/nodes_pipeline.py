"""Loading the model, sampling a trajectory, and looking at what came out.

The vertical slice of TF-diag Part 3's sampling block: TF Load Pipeline holds the
frozen DiT and the RAE decoder, TF Generate walks the coarse-to-fine loop once,
and TF Decode / TF Latent Preview are the two ways to inspect the result.
"""
from __future__ import annotations

from comfy_api.latest import io

from . import render
from .data import LevelStack
from .locate import AUTO_CHECKPOINT, imagenet_classes, list_checkpoints, tf_configs
from .pipeline import load_pipeline
from .sockets import (
    CATEGORY,
    TFLevelsSocket,
    TFPipelineSocket,
    level_input,
    pipeline_input,
    resolve_pipeline,
    seed_input,
)

WHICH_LEVELS = ["all levels", "final level only", "single level"]

_CLASS_CHOICES: list[str] | None = None


def class_choices() -> list[str]:
    """`"213 - Irish setter, red setter"` for each ImageNet-1k id, cached."""
    global _CLASS_CHOICES
    if _CLASS_CHOICES is None:
        classes = imagenet_classes()
        _CLASS_CHOICES = [f"{i} - {classes[i]}" for i in sorted(classes)]
    return _CLASS_CHOICES


def which_input() -> io.Combo.Input:
    return io.Combo.Input(
        "which",
        options=WHICH_LEVELS,
        tooltip="Which levels of the trajectory to render.",
    )


def pick_levels(levels: LevelStack, which: str, level: int) -> list[int]:
    if which == "final level only":
        return [levels.num_levels - 1]
    if which == "single level":
        return [levels.clamp(level)]
    return list(range(levels.num_levels))


class TFLoadPipeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLoadPipeline",
            display_name="TF Load Pipeline",
            category=CATEGORY,
            description=(
                "Load a Trajectory Forcing flow model and its RAE decoder. The handle is cached "
                "for the life of the ComfyUI process, so re-queueing never re-reads the checkpoint."
            ),
            search_aliases=["trajectory forcing", "tf"],
            inputs=[
                io.Combo.Input(
                    "checkpoint",
                    options=list_checkpoints(),
                    default=AUTO_CHECKPOINT,
                    tooltip="Flow checkpoint from models/trajectory_forcing/. "
                            f"{AUTO_CHECKPOINT!r} fetches the released editing checkpoint into that folder.",
                ),
                io.Combo.Input(
                    "config",
                    options=tf_configs(),
                    tooltip="TrajectoryForcing config. edit_env_config.yml is the one that matches "
                            "the TF_L_edit checkpoint; the others are training/eval configs.",
                ),
                io.Boolean.Input(
                    "warmup",
                    default=True,
                    tooltip="Sample and decode one throwaway image at load, paying the XLA compile "
                            "and the decoder build here instead of on the first real prompt.",
                ),
            ],
            outputs=[
                TFPipelineSocket.Output("pipeline"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, checkpoint, config, warmup) -> io.NodeOutput:
        pipe = load_pipeline(config, checkpoint, warmup)
        return io.NodeOutput(pipe, pipe.describe())


class TFImageNetClass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choices = class_choices()
        return io.Schema(
            node_id="TFImageNetClass",
            display_name="TF ImageNet Class",
            category=CATEGORY,
            description="Pick an ImageNet-1k class by name and get its id.",
            inputs=[io.Combo.Input("class_name", options=choices, default=choices[0])],
            outputs=[io.Int.Output("class_id"), io.String.Output("name")],
        )

    @classmethod
    def execute(cls, class_name) -> io.NodeOutput:
        class_id, name = class_name.split(" - ", 1)
        return io.NodeOutput(int(class_id), name)


class TFGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFGenerate",
            display_name="TF Generate",
            category=CATEGORY,
            description=(
                "Sample a full coarse-to-fine trajectory: one network evaluation per level, each "
                "level conditioned on the level below it. Outputs every level, not just the last."
            ),
            inputs=[
                TFPipelineSocket.Input("pipeline"),
                io.Int.Input(
                    "class_id", default=213, min=0, max=999,
                    tooltip="ImageNet-1k class to condition on. Wire TF ImageNet Class in to pick by name.",
                ),
                seed_input(),
            ],
            outputs=[TFLevelsSocket.Output("levels")],
        )

    @classmethod
    def execute(cls, pipeline, class_id, seed) -> io.NodeOutput:
        latents = pipeline.generate(class_id, seed)
        return io.NodeOutput(
            LevelStack(
                latents=latents,
                class_id=int(class_id),
                seed=int(seed),
                history=(f"generate(class={int(class_id)}, seed={int(seed)})",),
                pipeline=pipeline,
            )
        )


class TFDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFDecode",
            display_name="TF Decode Levels",
            category=CATEGORY,
            description=(
                "Run the RAE decoder on the trajectory. Because the decoder is frozen and works on "
                "any point in DINOv2 space, every intermediate level decodes, not only the last."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                which_input(),
                level_input(
                    tooltip="Level to decode. IGNORED unless 'which' is 'single level'."),
                io.Boolean.Input(
                    "label_levels", default=True,
                    tooltip="Write 'Level 2 (subparts)' under each frame, since a batch "
                            "preview shows no captions.",
                ),
                pipeline_input(),
            ],
            outputs=[io.Image.Output("images"), io.String.Output("warnings")],
        )

    @classmethod
    def execute(cls, levels, which, level, label_levels, pipeline=None) -> io.NodeOutput:
        pipeline = resolve_pipeline(pipeline, levels, "TF Decode Levels")
        wanted = pick_levels(levels, which, level)
        # decode_last is a real saving on the ViT-XL decoder; anything else goes
        # through decode_all and is subset afterwards.
        if wanted == [levels.num_levels - 1]:
            frames = pipeline.decode(levels.latents, final_only=True)
        else:
            decoded = pipeline.decode(levels.latents, final_only=False)
            frames = [decoded[i] for i in wanted]
        if label_levels:
            frames = [
                render.caption(f, render.level_caption(i, levels.num_levels))
                for f, i in zip(frames, wanted, strict=True)
            ]
        warning = ""
        if levels.dirty_level is not None and max(wanted) > levels.dirty_level:
            warning = (
                f"Levels above {levels.dirty_level} are stale: the canvas at level "
                f"{levels.dirty_level} was edited but TF Resume From Level has not run yet."
            )
        return io.NodeOutput(render.to_image(frames), warning)


class TFLatentPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLatentPreview",
            display_name="TF Latent Preview (PCA)",
            category=CATEGORY,
            description=(
                "Show the raw token grid as a PCA false-colour image -- far cheaper than decoding, "
                "and it shows the region structure the edit nodes operate on."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                which_input(),
                level_input(tooltip="Level to render. IGNORED unless 'which' is 'single level'."),
                io.Int.Input(
                    "size", default=512, min=64, max=2048, step=64,
                    tooltip="Approximate output width; rounded so each token is a whole "
                            "number of pixels.",
                ),
                io.Boolean.Input("label_levels", default=True),
                TFLevelsSocket.Input(
                    "palette_from", optional=True,
                    tooltip="Fit the PCA colours jointly with this trajectory too, so two images "
                            "being compared are coloured on the same axes.",
                ),
                pipeline_input(),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, levels, which, level, size, label_levels,
                palette_from=None, pipeline=None) -> io.NodeOutput:
        pipeline = resolve_pipeline(pipeline, levels, "TF Latent Preview")
        palette = None
        if palette_from is not None:
            palette = pipeline.fit_palette([levels.latents, palette_from.latents])
        tiles = pipeline.pca_tiles(levels.latents, palette=palette)
        wanted = pick_levels(levels, which, level)
        frames = [render.fit_to_grid(tiles[i], levels.grid, size) for i in wanted]
        if label_levels:
            frames = [
                render.caption(f, render.level_caption(i, levels.num_levels))
                for f, i in zip(frames, wanted, strict=True)
            ]
        return io.NodeOutput(render.to_image(frames))


class TFLevelsInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLevelsInfo",
            display_name="TF Levels Info",
            category=CATEGORY,
            description="Shape, class, seed and the edit history of a trajectory.",
            inputs=[TFLevelsSocket.Input("levels")],
            outputs=[
                io.String.Output("info"),
                io.Int.Output("num_levels"),
                io.Int.Output("class_id"),
                io.String.Output("class_name"),
                io.Int.Output("seed"),
            ],
        )

    @classmethod
    def execute(cls, levels) -> io.NodeOutput:
        name = imagenet_classes().get(int(levels.class_id), "")
        return io.NodeOutput(
            levels.describe(), levels.num_levels, levels.class_id, name, levels.seed)
