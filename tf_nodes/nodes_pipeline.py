"""Loading the model, sampling a trajectory, and looking at what came out.

The vertical slice of TF-diag Part 3's sampling block: TF Load Pipeline holds the
frozen DiT and the RAE decoder, TF Generate walks the coarse-to-fine loop once,
and TF Decode / TF Latent Preview are the two ways to inspect the result.
"""
from __future__ import annotations

import time

from comfy_api.latest import io

from . import render
from .data import LevelStack
from .locate import AUTO_CHECKPOINT, imagenet_classes, list_checkpoints, tf_configs
from .pipeline import load_pipeline
from .sockets import (
    CATEGORY_GENERATE,
    SHIPPED_LEVELS,
    TFLevelsSocket,
    TFPipelineSocket,
    auto_level_input,
    node_preview,
    pipeline_input,
    progress_bar,
    resolve_pipeline,
    seed_input,
    sheet_layout_input,
)

# The class both TF Generate and TF ImageNet Class start on, so a fresh graph
# does not disagree with itself about what a default run produces.
DEFAULT_CLASS = 213  # Irish setter

# One dropdown instead of a mode plus a level that the mode silently ignored.
WHICH_LEVELS = (["all levels", "final level only"]
                + [f"level {i}" for i in range(SHIPPED_LEVELS)])

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
        tooltip="Which levels to render. Naming a level here replaces the old "
                "mode-plus-number pair, which left a number that did nothing in "
                "two of its three modes.",
    )


def level_override_input() -> io.Int.Input:
    return auto_level_input(
        "level_override", "follows the dropdown above",
        f"Set a level number only for a model with more than {SHIPPED_LEVELS} levels, "
        "which no released checkpoint has. It wins over the dropdown when set.",
    )


def pick_levels(levels: LevelStack, which: str, level_override: int) -> list[int]:
    """The levels to render. Never ignores either input: the override always wins
    when it is set, and the dropdown decides otherwise."""
    if int(level_override) >= 0:
        return [levels.clamp(level_override)]
    if which == "final level only":
        return [levels.num_levels - 1]
    if which == "all levels":
        return list(range(levels.num_levels))
    return [levels.clamp(int(which.rsplit(" ", 1)[1]))]


class TFLoadPipeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLoadPipeline",
            has_intermediate_output=True,
            search_aliases=["trajectory forcing", "tf", "checkpoint", "model", "loader"],
            display_name="TF Load Pipeline",
            category=CATEGORY_GENERATE,
            description=(
                "Load a Trajectory Forcing flow model and its RAE decoder. The handle is cached "
                "for the life of the ComfyUI process, so re-queueing never re-reads the checkpoint."
            ),
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
                    options=tf_configs(), advanced=True,
                    tooltip="TrajectoryForcing config. edit_env_config.yml is the one that matches "
                            "the TF_L_edit checkpoint; the others are training/eval configs.",
                ),
                io.Boolean.Input(
                    "warmup",
                    default=True, advanced=True,
                    tooltip="Sample and decode one throwaway image at load, paying the XLA compile "
                            "and the decoder build here -- where this node's progress bar shows "
                            "them -- instead of on your first real prompt.\n\n"
                            "Turning it off does not save the 1-2 minutes, it moves them to the "
                            "first TF Generate or TF Decode, which cannot show progress for a "
                            "single opaque compile. Leave it on unless you are loading the model "
                            "without intending to sample.",
                ),
            ],
            outputs=[
                TFPipelineSocket.Output("pipeline"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, checkpoint, config, warmup) -> io.NodeOutput:
        # The slowest thing in the extension by a wide margin, and until now the
        # only slow thing with nothing to watch: a first run sat for one to two
        # minutes with an idle graph, which is indistinguishable from a hung one
        # -- and was reported as exactly that.
        bar = progress_bar(3)
        # ProgressBar sends nothing when it is constructed -- its first message
        # goes out on the first `update`. That left the whole checkpoint restore
        # (seconds warm, most of the wait cold) with no bar on screen at all,
        # which is the part that most needs one. Send 0/3 up front so it appears
        # immediately rather than a third of the way in.
        if bar:
            bar.update_absolute(0, 3)
        stages: list[str] = []

        def on_step(label: str) -> None:
            stages.append(label)
            if bar:
                bar.update(1)

        started = time.perf_counter()
        pipe = load_pipeline(config, checkpoint, warmup, on_step)
        report = f"{pipe.describe()}\nready in {time.perf_counter() - started:.1f}s"
        if stages:
            report += "\n" + "; ".join(stages)
        return io.NodeOutput(pipe, report, ui=node_preview(text=report))


class TFImageNetClass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choices = class_choices()
        return io.Schema(
            node_id="TFImageNetClass",
            has_intermediate_output=True,
            search_aliases=["class", "imagenet", "label", "category", "dog", "cat"],
            display_name="TF ImageNet Class",
            category=CATEGORY_GENERATE,
            description="Pick an ImageNet-1k class by name and get its id.",
            inputs=[io.Combo.Input(
                "class_name", options=choices, default=choices[DEFAULT_CLASS],
                tooltip="Wire the class_id output into TF Generate.",
            )],
            outputs=[io.Int.Output("class_id"), io.String.Output("name")],
        )

    @classmethod
    def execute(cls, class_name) -> io.NodeOutput:
        class_id, name = class_name.split(" - ", 1)
        return io.NodeOutput(int(class_id), name, ui=node_preview(text=f"{class_id}: {name}"))


class TFGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFGenerate",
            search_aliases=["sample", "trajectory", "seed", "txt2img", "generate"],
            display_name="TF Generate",
            category=CATEGORY_GENERATE,
            description=(
                "Sample a full coarse-to-fine trajectory: one network evaluation per level, each "
                "level conditioned on the level below it. Outputs every level, not just the last."
            ),
            inputs=[
                TFPipelineSocket.Input("pipeline"),
                io.Int.Input(
                    "class_id", default=DEFAULT_CLASS, min=0, max=999,
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
            search_aliases=["decode", "rae", "image", "pixels", "vae"],
            display_name="TF Decode Levels",
            category=CATEGORY_GENERATE,
            has_intermediate_output=True,
            description=(
                "Run the RAE decoder on the trajectory. Because the decoder is frozen and works on "
                "any point in DINOv2 space, every intermediate level decodes, not only the last."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                which_input(),
                io.Boolean.Input(
                    "label_levels", default=True, advanced=True,
                    tooltip="Write 'Level 2 (subparts)' under each frame, since a batch "
                            "preview shows no captions.",
                ),
                sheet_layout_input("the decoded levels"),
                level_override_input(),
                pipeline_input(),
            ],
            outputs=[io.Image.Output("images"), io.String.Output("warnings")],
        )

    @classmethod
    def execute(cls, levels, which, label_levels, sheet_layout, level_override,
                pipeline=None) -> io.NodeOutput:
        pipeline = resolve_pipeline(pipeline, levels, "TF Decode Levels")
        wanted = pick_levels(levels, which, level_override)
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
        notes = []
        if levels.dirty_level is not None and max(wanted) > levels.dirty_level:
            notes.append(
                f"Levels above {levels.dirty_level} are stale: the canvas at level "
                f"{levels.dirty_level} was edited but TF Resume From Level has not run yet."
            )
        warning = "\n".join(n for n in notes if n)
        shown = ", ".join(str(i) for i in wanted)
        return io.NodeOutput(
            render.lay_out(frames, sheet_layout), warning,
            ui=node_preview(text=f"decoded level(s) {shown}" + (f"\n{warning}" if warning else "")),
        )


class TFLatentPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLatentPreview",
            search_aliases=["latent", "pca", "tokens", "preview", "false colour"],
            display_name="TF Latent Preview (PCA)",
            category=CATEGORY_GENERATE,
            has_intermediate_output=True,
            description=(
                "Show the raw token grid as a PCA false-colour image -- far cheaper than decoding, "
                "and it shows the region structure the edit nodes operate on."
            ),
            inputs=[
                TFLevelsSocket.Input("levels"),
                which_input(),
                io.Int.Input(
                    "size", default=512, min=64, max=2048, step=64, advanced=True,
                    tooltip="Approximate output width; rounded so each token is a whole "
                            "number of pixels.",
                ),
                io.Boolean.Input("label_levels", default=True, advanced=True),
                sheet_layout_input("the level tiles"),
                level_override_input(),
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
    def execute(cls, levels, which, size, label_levels, sheet_layout, level_override,
                palette_from=None, pipeline=None) -> io.NodeOutput:
        pipeline = resolve_pipeline(pipeline, levels, "TF Latent Preview")
        palette = None
        if palette_from is not None:
            palette = pipeline.fit_palette([levels.latents, palette_from.latents])
        tiles = pipeline.pca_tiles(levels.latents, palette=palette)
        wanted = pick_levels(levels, which, level_override)
        frames = [render.fit_to_grid(tiles[i], levels.grid, size) for i in wanted]
        if label_levels:
            frames = [
                render.caption(f, render.level_caption(i, levels.num_levels))
                for f, i in zip(frames, wanted, strict=True)
            ]
        shown = ", ".join(str(i) for i in wanted)
        return io.NodeOutput(
            render.lay_out(frames, sheet_layout), ui=node_preview(text=f"level(s) {shown}"))


class TFLevelsInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLevelsInfo",
            search_aliases=["info", "seed", "class", "history", "metadata"],
            display_name="TF Levels Info",
            category=CATEGORY_GENERATE,
            has_intermediate_output=True,
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
        report = f"class {levels.class_id}: {name}\n{levels.describe()}" if name else levels.describe()
        return io.NodeOutput(
            levels.describe(), levels.num_levels, levels.class_id, name, levels.seed,
            ui=node_preview(text=report))
