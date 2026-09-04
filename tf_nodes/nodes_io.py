"""Getting a trajectory on and off disk.

A trajectory costs GPU time to sample and is the thing every edit is measured
against, so being able to reload the exact one an earlier run used -- across a
ComfyUI restart, or in a different workflow -- is what makes a comparison
between two edits mean anything.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from comfy_api.latest import io

from .data import LevelStack
from .locate import imagenet_classes
from .sockets import CATEGORY_IO, TFLevelsSocket, node_preview

SUBDIR = "trajectory_forcing"


def _output_dir() -> Path:
    import folder_paths

    path = Path(folder_paths.get_output_directory()) / SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_saved() -> list[str]:
    try:
        return sorted(p.name for p in _output_dir().glob("*.npz"))
    except Exception:
        return []


class TFSaveLevels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFSaveLevels",
            search_aliases=["save", "export", "npz", "store", "keep"],
            has_intermediate_output=True,
            display_name="TF Save Levels",
            category=CATEGORY_IO,
            description=f"Write a trajectory to output/{SUBDIR}/<name>.npz, "
                        "with its class, seed and edit history.",
            is_output_node=True,
            inputs=[
                TFLevelsSocket.Input("levels"),
                io.String.Input("name", default="trajectory"),
                io.Boolean.Input(
                    "overwrite", default=False, advanced=True,
                    tooltip="Off appends -001, -002, ... rather than replacing an existing file.",
                ),
            ],
            outputs=[io.String.Output("path")],
        )

    @classmethod
    def execute(cls, levels, name, overwrite) -> io.NodeOutput:
        stem = "".join(c for c in name.strip() if c.isalnum() or c in "-_.") or "trajectory"
        path = _output_dir() / f"{stem}.npz"
        if not overwrite:
            n = 1
            while path.exists():
                path = _output_dir() / f"{stem}-{n:03d}.npz"
                n += 1
        np.savez_compressed(
            path,
            latents=levels.latents,
            class_id=np.int32(levels.class_id),
            seed=np.int64(levels.seed),
            history=np.array(levels.history, dtype=object),
            dirty_level=np.int32(-1 if levels.dirty_level is None else levels.dirty_level),
        )
        return io.NodeOutput(str(path), ui=node_preview(text=f"wrote {path}"))


class TFSaveReport(io.ComfyNode):
    """The one output of this extension that had nowhere to go.

    TF Sweep Edit and TF Compare Levels produce exactly the numbers a writeup
    wants, and until this node they existed only as text in a node body you had
    to select and copy. So the tool could measure an edit and not archive the
    measurement -- which, by this repo's own rule that a number whose run cannot
    be traced is worth much less than it looks, is the wrong half to be missing.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFSaveReport",
            search_aliases=["save", "report", "table", "results", "notes", "text", "markdown"],
            has_intermediate_output=True,
            display_name="TF Save Report",
            category=CATEGORY_IO,
            is_output_node=True,
            description=(
                f"Write a report to output/{SUBDIR}/<name>.md, so the numbers a run produced "
                "outlive the browser tab.\n\n"
                "Wire TF Sweep Edit's or TF Compare Levels' 'report' into it. Wire the "
                "trajectory in too and the file carries the class, seed and full edit history "
                "that produced those numbers -- a table nobody can trace back to a run is "
                "worth much less than it looks."
            ),
            inputs=[
                io.String.Input(
                    "text", default="", multiline=True, force_input=True,
                    tooltip="The report. Any STRING output works; the sweep and compare tables "
                            "are what this exists for.",
                ),
                io.String.Input("name", default="report"),
                TFLevelsSocket.Input(
                    "levels", optional=True,
                    tooltip="Optional provenance: class, seed and edit history are written above "
                            "the report so the numbers can be traced back to the run.",
                ),
                io.Boolean.Input(
                    "append", default=True, advanced=True,
                    tooltip="Add to the end of an existing file under a new timestamped heading, "
                            "so one session's runs accumulate into one comparable table. Off "
                            "writes <name>-001.md, -002.md, ... instead.",
                ),
            ],
            outputs=[io.String.Output("path")],
        )

    @classmethod
    def execute(cls, text, name, levels=None, append=True) -> io.NodeOutput:
        import time

        if not (text or "").strip():
            raise ValueError(
                "'text' is empty, so there is nothing to save. Wire TF Sweep Edit's or "
                "TF Compare Levels' 'report' output into it."
            )
        stem = "".join(c for c in name.strip() if c.isalnum() or c in "-_.") or "report"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        path = _output_dir() / f"{stem}.md"
        if not append:
            n = 1
            while path.exists():
                path = _output_dir() / f"{stem}-{n:03d}.md"
                n += 1

        lines = [f"## {stamp}", ""]
        if levels is not None:
            named = imagenet_classes().get(int(levels.class_id), "")
            lines += [
                f"- class: {levels.class_id}" + (f" ({named})" if named else ""),
                f"- seed: {levels.seed}",
                f"- levels: {levels.num_levels}, grid {levels.grid[0]}x{levels.grid[1]}",
                "- history:",
                *(f"  {i + 1}. {h}" for i, h in enumerate(levels.history)),
                "",
            ]
        # Fenced, because these reports are aligned columns and markdown would
        # otherwise reflow them into one paragraph -- which is the whole table.
        lines += ["```", text.rstrip(), "```", ""]
        block = "\n".join(lines)

        existing = path.exists() and path.stat().st_size > 0
        with path.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(("\n" + block) if (append and existing) else block)
        note = f"{'appended to' if append else 'wrote'} {path}"
        return io.NodeOutput(str(path), ui=node_preview(text=note))


class TFSaveImages(io.ComfyNode):
    """The pictures, next to the latents and the numbers.

    `TF Save Levels` archives latents and `TF Save Report` archives the table;
    what a run that has to be cited later still lost was the images someone
    actually looked at. Every workflow here ends in `PreviewImage`, which writes
    to ComfyUI's **temp** directory -- so the pictures are one cache-clear from
    gone.

    For a single trajectory that is an inconvenience: the `.npz` has the latents,
    so `TF Load Levels` -> `TF Decode` reproduces the frames exactly, at the cost
    of a GPU and a model load. For a **sweep** it is a real loss. Only
    `output_arm`'s trajectory leaves the node on the `levels` socket; every other
    arm's latents are discarded when `execute` returns, so those frames exist
    nowhere but the contact sheet. Re-running with the same seeds would rebuild
    them, but that is redoing the experiment rather than reading its record.

    Core's `SaveImage` writes PNGs perfectly well and is not being replaced for
    the sake of it. What it cannot do is tie the files to the run: it names them
    `ComfyUI_00001_.png` in the output root, so a trajectory, its table and its
    pictures land under three unrelated names. This shares the `name` widget with
    the other two save nodes, writes beside them in output/{SUBDIR}/, and stamps
    the class, seed and edit history into each PNG's own metadata -- which is
    what makes a picture in a writeup traceable back to the run that made it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFSaveImages",
            search_aliases=["save", "png", "image", "sheet", "export", "archive", "keep"],
            has_intermediate_output=True,
            display_name="TF Save Images",
            category=CATEGORY_IO,
            is_output_node=True,
            description=(
                f"Write images to output/{SUBDIR}/<name>.png, beside the trajectory and the "
                "report from the same run.\n\n"
                "PreviewImage writes to ComfyUI's temp directory, so what you looked at does "
                "not survive a cache clear. A sweep's contact sheet is the pressing case: only "
                "one arm's latents leave the node, so the other arms exist nowhere else.\n\n"
                "Wire the trajectory in too and each PNG carries the class, seed and edit "
                "history in its metadata."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Any IMAGE. A batch is written as one file per frame; a stitched "
                            "contact sheet is one file.",
                ),
                io.String.Input("name", default="images"),
                TFLevelsSocket.Input(
                    "levels", optional=True,
                    tooltip="Optional provenance: class, seed and edit history are written into "
                            "each PNG's metadata so a picture can be traced back to its run.",
                ),
                io.Boolean.Input(
                    "overwrite", default=False, advanced=True,
                    tooltip="Off appends -001, -002, ... rather than replacing existing files.",
                ),
            ],
            outputs=[io.String.Output("path")],
        )

    @classmethod
    def execute(cls, images, name, levels=None, overwrite=False) -> io.NodeOutput:
        from PIL import Image, PngImagePlugin

        # Tensor or array, the way render.from_mask takes either: this module
        # otherwise never imports torch and there is no reason to start.
        arr = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
        frames = np.round(np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0) * 255.0)
        frames = frames.astype(np.uint8)
        if frames.ndim == 3:  # a bare HWC image rather than a batch
            frames = frames[None]
        # A guard rather than an IndexError three lines down: an empty batch is a
        # degenerate result worth reporting, and this repo has already paid for
        # letting one reach a library that raises instead.
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise ValueError(
                f"Expected an IMAGE batch of shape [B,H,W,C] with at least one frame, "
                f"got {tuple(np.shape(images))}."
            )

        stem = "".join(c for c in name.strip() if c.isalnum() or c in "-_.") or "images"
        count = int(frames.shape[0])

        def names(base: str) -> list[str]:
            # One frame keeps the bare stem, so the common case is `sweep.png`
            # rather than `sweep-01.png` and matches `sweep.npz` / `sweep.md`.
            return ([f"{base}.png"] if count == 1
                    else [f"{base}-{i + 1:02d}.png" for i in range(count)])

        chosen, n = stem, 1
        while not overwrite and any((_output_dir() / f).exists() for f in names(chosen)):
            chosen, n = f"{stem}-{n:03d}", n + 1

        meta = PngImagePlugin.PngInfo()
        if levels is not None:
            named = imagenet_classes().get(int(levels.class_id), "")
            meta.add_text("tf_class", f"{levels.class_id}" + (f" ({named})" if named else ""))
            meta.add_text("tf_seed", str(levels.seed))
            meta.add_text("tf_history", " | ".join(levels.history))

        written = []
        for frame, filename in zip(frames, names(chosen), strict=True):
            path = _output_dir() / filename
            Image.fromarray(frame[:, :, :3] if frame.shape[-1] >= 3 else frame[:, :, 0]).save(
                path, pnginfo=meta)
            written.append(path)

        note = f"wrote {len(written)} image{'s' if len(written) != 1 else ''} to {_output_dir()}"
        note += "\n" + "\n".join(p.name for p in written)
        if levels is None:
            note += "\n(no trajectory wired -- the files carry no class/seed metadata)"
        return io.NodeOutput(str(written[0]), ui=node_preview(text=note))


class TFLoadLevels(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TFLoadLevels",
            search_aliases=["load", "import", "npz", "restore", "reopen"],
            has_intermediate_output=True,
            display_name="TF Load Levels",
            category=CATEGORY_IO,
            description=f"Read back a trajectory saved by TF Save Levels from output/{SUBDIR}/.",
            inputs=[
                io.Combo.Input(
                    "file", options=list_saved() or ["<none saved yet>"],
                    tooltip="Trajectories under output/" + SUBDIR + "/. The list is built when "
                            "ComfyUI sends its node definitions, so one saved since this page "
                            "loaded appears after a refresh (press R).",
                ),
                io.String.Input(
                    "path_override", default="", advanced=True,
                    tooltip="Absolute path to an .npz elsewhere; overrides the dropdown.",
                ),
            ],
            outputs=[TFLevelsSocket.Output("levels"), io.String.Output("info")],
        )

    @classmethod
    def execute(cls, file, path_override) -> io.NodeOutput:
        path = Path(path_override.strip()) if path_override.strip() else _output_dir() / file
        if not path.is_file():
            raise FileNotFoundError(f"No saved trajectory at {path}")
        with np.load(path, allow_pickle=True) as data:
            dirty = int(data["dirty_level"])
            levels = LevelStack(
                latents=data["latents"],
                class_id=int(data["class_id"]),
                seed=int(data["seed"]),
                history=tuple(str(h) for h in data["history"].tolist()) + (f"loaded from {path.name}",),
                dirty_level=None if dirty < 0 else dirty,
            )
        return io.NodeOutput(levels, levels.describe(),
                             ui=node_preview(text=levels.describe()))
