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

from .tf_nodes import health, nodes
from .tf_nodes.locate import register_model_folder
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
        # Nothing in here may raise. An exception out of on_load takes the whole
        # extension with it: all 21 nodes disappear and ComfyUI-Manager reports
        # `IMPORT FAILED` with no cause, while the actual explanation goes to a
        # server console that someone driving a browser never reads. Locating
        # the checkout can involve a network fetch, so it is the most likely
        # thing here to fail and the least likely to be the user's fault --
        # which is exactly the wrong combination for a hard failure.
        report = health.report_at_startup()

        applied = configure_jax_env()
        models = register_model_folder()
        log.info(
            "TrajectoryForcing: repo=%s models=%s jax env set: %s",
            report.repo, models, ", ".join(applied) or "nothing (already exported)",
        )

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return nodes()


async def comfy_entrypoint() -> ComfyExtension:
    return TrajectoryForcingExtension()
