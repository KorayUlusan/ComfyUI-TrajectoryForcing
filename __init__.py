"""ComfyUI-TrajectoryForcing -- Trajectory Forcing's coarse-to-fine generation and
interactive latent editing as ComfyUI nodes.

ComfyUI calls `comfy_entrypoint` once at startup. Two things have to happen
before any node runs, and both are done here rather than inside a node:

* the JAX memory settings, because JAX initialises its GPU backend on first
  touch and there is no putting the environment back afterwards;
* registering models/trajectory_forcing/, because the loader node's checkpoint
  dropdown is built while its schema is defined.
"""
from __future__ import annotations

import logging

from comfy_api.latest import ComfyExtension, io

from .tf_nodes import nodes
from .tf_nodes.locate import register_model_folder, tf_repo
from .tf_nodes.tf_import import configure_jax_env

log = logging.getLogger(__name__)

# Served at /extensions/<dir>/ and loaded by the browser. One file: the
# clickable token grid on TF Tokens From Coords. It decorates that node's
# `coords` text field and is never the only way to set a selection -- if it
# fails to load, typing still works, which is the condition on this repo having
# any frontend code at all.
WEB_DIRECTORY = "./web"


class TrajectoryForcingExtension(ComfyExtension):
    async def on_load(self) -> None:
        repo = tf_repo()
        applied = configure_jax_env()
        models = register_model_folder()
        log.info(
            "TrajectoryForcing: repo=%s models=%s jax env set: %s",
            repo, models, ", ".join(applied) or "nothing (already exported)",
        )

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return nodes()


async def comfy_entrypoint() -> ComfyExtension:
    return TrajectoryForcingExtension()
