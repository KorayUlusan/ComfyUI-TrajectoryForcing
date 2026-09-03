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
                io.Combo.Input("file", options=list_saved() or ["<none saved yet>"]),
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
