#!/usr/bin/env python3
"""Generate the example workflows in workflows/ from the live node schemas.

Hand-written workflow JSON goes stale the moment a widget is added or reordered,
and the failure is silent: LiteGraph stores widget values positionally, so an
extra widget shifts every value after it and the workflow loads with plausible
wrong numbers rather than an error. Generating from `define_schema()` means the
positions cannot drift.

    PYTHONPATH=$COMFY_DIR python scripts/make_workflows.py

Emits two files per workflow:

  workflows/<name>.json      LiteGraph format -- what "Open" in the ComfyUI menu
                             expects, and what a user actually loads.
  workflows/api/<name>.json  API format -- what POST /prompt expects. This is
                             the one scripts/server_smoke.py executes, so it is
                             the format that gets tested rather than eyeballed.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = EXT_ROOT / "workflows"


def comfy_root() -> Path:
    env = os.environ.get("COMFY_DIR", "").strip()
    for path in ([Path(env)] if env else []) + [EXT_ROOT.parent.parent / "ComfyUI", Path.home() / "ComfyUI"]:
        if (path / "nodes.py").is_file():
            return path
    raise SystemExit("Could not find ComfyUI; set COMFY_DIR.")


sys.path.insert(0, str(comfy_root()))

# Node box geometry. LiteGraph recomputes sizes on load, so these only need to
# be roughly right for the initial layout to be readable.
COLUMN = 380
ROW = 60


class Graph:
    """Collects nodes and links, then serialises to both workflow formats."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.nodes: list[dict] = []
        self.links: list[list] = []
        # The API payload is built alongside the LiteGraph one rather than
        # derived from it afterwards: reconstructing which slot a value came
        # from means re-deriving the widget/socket split a second time, and the
        # two derivations drifting apart is exactly the bug this generator is
        # meant to prevent.
        self.api: dict[str, dict] = {}

    def add(self, node_type: str, column: int, row: int, title: str = "", **values) -> int:
        """Place a node. `values` sets widget values by input id and wires links.

        A value of `(node_id, output_index)` is a link; anything else is a widget
        value. Unset widgets keep the schema default. A widget input given a link
        becomes a connected socket, which is what LiteGraph calls a converted
        widget -- it keeps its slot in `widgets_values` and gains an entry in
        `inputs` carrying the widget's name.
        """
        spec = describe(node_type)
        node_id = len(self.nodes) + 1

        inputs, widgets, api_inputs = [], [], {}
        for input_ in spec.inputs:
            value = values.pop(input_.id, _UNSET)
            linked = _is_link(value)
            if input_.is_widget:
                widgets.extend(_widget_values(input_, _UNSET if linked else value))
                if not linked:
                    api_inputs[input_.id] = widgets[-2] if input_.control_after_generate else widgets[-1]
                    continue
            elif value is _UNSET and not input_.optional:
                raise ValueError(f"{node_type}.{input_.id} is required but nothing is wired to it")

            link = None
            if linked:
                origin, slot = value
                link = len(self.links) + 1
                self.links.append([link, origin, slot, node_id, len(inputs), input_.io_type])
                self.nodes[origin - 1]["outputs"][slot]["links"].append(link)
                api_inputs[input_.id] = [str(origin), slot]
            inputs.append({
                "label": input_.label,
                "name": input_.id,
                "type": input_.io_type,
                "link": link,
                **({"widget": {"name": input_.id}} if input_.is_widget else {}),
            })
        if values:
            raise KeyError(f"{node_type} has no input(s) {sorted(values)}")

        self.api[str(node_id)] = {
            "class_type": node_type,
            "inputs": api_inputs,
            "_meta": {"title": title or spec.title},
        }
        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "pos": [80 + column * COLUMN, 80 + row * ROW],
            "size": [COLUMN - 60, 40 + 26 * (len(inputs) + len(widgets))],
            "flags": {},
            "order": node_id - 1,
            "mode": 0,
            "inputs": inputs,
            "outputs": [
                {
                    "label": out.label,
                    "name": out.label,
                    "type": out.io_type,
                    "slot_index": i,
                    "links": [],
                }
                for i, out in enumerate(spec.outputs)
            ],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets,
            **({"title": title} if title else {}),
        })
        return node_id

    def as_workflow(self) -> dict:
        return {
            "id": f"trajectory-forcing-{self.name}",
            "revision": 0,
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {"description": self.description},
            "version": 0.4,
        }

    def write(self) -> None:
        (WORKFLOWS / "api").mkdir(parents=True, exist_ok=True)
        (WORKFLOWS / f"{self.name}.json").write_text(json.dumps(self.as_workflow(), indent=2) + "\n")
        (WORKFLOWS / "api" / f"{self.name}.json").write_text(json.dumps(self.api, indent=2) + "\n")
        print(f"wrote {self.name}: {len(self.nodes)} nodes, {len(self.links)} links")


_UNSET = object()


def _is_link(value) -> bool:
    """`(node_id, output_slot)` is a link; every widget value in these workflows
    is a scalar or a string, so the shape is unambiguous."""
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int)


# Widgets rather than sockets, for a V1 node whose INPUT_TYPES gives only a type
# string. A list type is a combo; these four are the scalar widgets. Everything
# else (IMAGE, MASK, MODEL, TF_*) is a socket.
_V1_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


@dataclass
class InputSpec:
    id: str
    io_type: str
    is_widget: bool
    optional: bool = False
    default: object = None
    options: list | None = None
    control_after_generate: bool = False
    label: str = ""


@dataclass
class OutputSpec:
    id: str
    io_type: str
    label: str = ""


@dataclass
class NodeSpec:
    node_type: str
    title: str
    inputs: list[InputSpec]
    outputs: list[OutputSpec]


_SPECS: dict[str, NodeSpec] = {}


def describe(node_type: str) -> NodeSpec:
    """Normalise a node class to inputs/outputs, whether it is V3 or legacy V1.

    Core ComfyUI is mid-migration: this extension's nodes define a V3 `Schema`,
    but PreviewImage -- which every one of these workflows ends in -- is still a
    V1 class with `INPUT_TYPES`. Both have to be readable or the generator can
    only wire its own nodes together.
    """
    if node_type in _SPECS:
        return _SPECS[node_type]
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(node_type)
    if cls is None:
        raise KeyError(f"ComfyUI has no node {node_type!r}")
    spec = _from_v3(cls) if hasattr(cls, "define_schema") else _from_v1(node_type, cls)
    _SPECS[node_type] = spec
    return spec


def _from_v3(cls) -> NodeSpec:
    schema = cls.define_schema()
    inputs = []
    for spec in schema.inputs:
        widget = hasattr(spec, "default")  # WidgetInput carries a default; Input does not
        inputs.append(InputSpec(
            id=spec.id,
            io_type=spec.get_io_type(),
            is_widget=widget,
            optional=bool(spec.optional),
            default=getattr(spec, "default", None),
            options=list(spec.options) if getattr(spec, "options", None) else None,
            control_after_generate=bool(getattr(spec, "control_after_generate", False)),
            label=spec.display_name or spec.id,
        ))
    outputs = [
        OutputSpec(id=out.id, io_type=out.get_io_type(), label=out.display_name or out.id)
        for out in schema.outputs
    ]
    return NodeSpec(schema.node_id, schema.display_name or schema.node_id, inputs, outputs)


def _from_v1(node_type: str, cls) -> NodeSpec:
    types = cls.INPUT_TYPES()
    inputs = []
    for section, optional in (("required", False), ("optional", True)):
        for name, declaration in (types.get(section) or {}).items():
            io_type = declaration[0] if isinstance(declaration, (list, tuple)) else declaration
            opts = declaration[1] if len(declaration) > 1 and isinstance(declaration[1], dict) else {}
            combo = isinstance(io_type, list)
            widget = (combo or io_type in _V1_WIDGET_TYPES) and not opts.get("forceInput")
            inputs.append(InputSpec(
                id=name,
                io_type="COMBO" if combo else io_type,
                is_widget=widget,
                optional=optional,
                default=opts.get("default", io_type[0] if combo and io_type else None),
                options=io_type if combo else None,
                control_after_generate=bool(opts.get("control_after_generate")),
                label=name,
            ))
    names = getattr(cls, "RETURN_NAMES", None) or getattr(cls, "RETURN_TYPES", ())
    outputs = [
        OutputSpec(id=name, io_type=io_type, label=name)
        for name, io_type in zip(names, getattr(cls, "RETURN_TYPES", ()), strict=False)
    ]
    return NodeSpec(node_type, getattr(cls, "DISPLAY_NAME", node_type), inputs, outputs)


def imagenet_option(class_id: int) -> str:
    """The TF ImageNet Class dropdown entry for an id, read from the node's own list.

    Spelling one out by hand does not survive contact with the server: the
    combo is validated against the live options, and a workflow naming a class
    string that is one word off is rejected before anything runs.
    """
    prefix = f"{class_id} - "
    for option in describe("TFImageNetClass").inputs[0].options or []:
        if option.startswith(prefix):
            return option
    raise KeyError(f"no ImageNet class option for id {class_id}")


def _widget_values(spec: InputSpec, value) -> list:
    resolved = spec.default if value is _UNSET else value
    if resolved is None and spec.options:
        resolved = spec.options[0]
    # ComfyUI validates every combo value against its option list and rejects
    # the whole prompt if one is off, so catch it here rather than at the server.
    if spec.options and resolved not in spec.options:
        raise ValueError(
            f"{resolved!r} is not one of {spec.id}'s {len(spec.options)} options"
            + (f"; did you mean {spec.options[0]!r}?" if len(spec.options) < 20 else "")
        )
    out = [resolved]
    if spec.control_after_generate:
        # LiteGraph inserts a companion widget after a seed; it occupies a slot
        # in widgets_values, and forgetting it shifts every later value.
        out.append("fixed")
    return out


# ---------------------------------------------------------------------------
# the workflows
# ---------------------------------------------------------------------------
def generate_and_decode() -> Graph:
    g = Graph(
        "01-generate-and-decode",
        "The vertical slice: sample one coarse-to-fine trajectory and look at every level, "
        "as decoded RGB and as the raw PCA token grid.",
    )
    pipe = g.add("TFLoadPipeline", 0, 0)
    cls = g.add("TFImageNetClass", 0, 5, class_name=imagenet_option(213))
    gen = g.add("TFGenerate", 1, 0, pipeline=(pipe, 0), class_id=(cls, 0), seed=592)
    dec = g.add("TFDecode", 2, 0, pipeline=(pipe, 0), levels=(gen, 0), which="all levels")
    pca = g.add("TFLatentPreview", 2, 6, pipeline=(pipe, 0), levels=(gen, 0), which="all levels")
    g.add("PreviewImage", 3, 0, title="Decoded RGB per level", images=(dec, 0))
    g.add("PreviewImage", 3, 5, title="Latent PCA per level", images=(pca, 0))
    g.add("TFLevelsInfo", 3, 10, levels=(gen, 0))
    return g


def feature_edit_coords() -> Graph:
    g = Graph(
        "02-feature-edit-coords",
        "Cross-image feature edit driven by typed token coordinates: take the mean feature of a "
        "region of one trajectory, write it into a region of another, and re-sample the finer "
        "levels. Runs as-is with no painting, so it is the reproducible version.",
    )
    pipe = g.add("TFLoadPipeline", 0, 0)
    target = g.add("TFGenerate", 1, 0, title="target", pipeline=(pipe, 0), class_id=213, seed=592)
    source = g.add("TFGenerate", 1, 5, title="source", pipeline=(pipe, 0), class_id=207, seed=592)

    regions = g.add("TFRegionMap", 2, 0, levels=(target, 0), level=2, cosine_threshold=0.9)
    tgt_tokens = g.add(
        "TFTokensFromCoords", 2, 6, title="target region",
        coords="7,7", levels=(target, 0), regions=(regions, 0),
    )
    src_tokens = g.add(
        "TFTokensFromCoords", 2, 10, title="source tokens",
        coords="6,6:9 7,6:9", levels=(source, 0),
    )

    edit = g.add(
        "TFFeatureEdit", 3, 0,
        levels=(target, 0), level=2, target_tokens=(tgt_tokens, 0), source_tokens=(src_tokens, 0),
        source_mode="region mean", strength=1.0, source_level=2, source_levels=(source, 0),
    )
    resume = g.add(
        "TFResumeFromLevel", 4, 0,
        pipeline=(pipe, 0), levels=(edit, 0), follow_edit=True, class_id=-1, seed=592,
    )

    before = g.add("TFDecode", 4, 8, title="before", pipeline=(pipe, 0), levels=(target, 0),
                   which="final level only")
    after = g.add("TFDecode", 5, 0, title="after", pipeline=(pipe, 0), levels=(resume, 0),
                  which="all levels")
    canvas = g.add("TFLevelCanvas", 3, 8, pipeline=(pipe, 0), levels=(target, 0), level=2,
                   view="latent PCA", regions=(regions, 0), highlight=(tgt_tokens, 0))
    g.add("PreviewImage", 6, 0, title="edited, all levels", images=(after, 0))
    g.add("PreviewImage", 6, 5, title="unedited final", images=(before, 0))
    g.add("PreviewImage", 4, 12, title="what is selected", images=(canvas, 0))
    return g


def feature_edit_painter() -> Graph:
    g = Graph(
        "03-feature-edit-painter",
        "The same edit, but the target region is painted. Run TF Level Canvas once, paint on the "
        "Painter node over the token grid it produces, then run again -- the mask is snapped to "
        "whole cosine regions, so a rough stroke still selects a clean part.",
    )
    pipe = g.add("TFLoadPipeline", 0, 0)
    target = g.add("TFGenerate", 1, 0, title="target", pipeline=(pipe, 0), class_id=213, seed=592)
    source = g.add("TFGenerate", 1, 5, title="source", pipeline=(pipe, 0), class_id=207, seed=592)
    regions = g.add("TFRegionMap", 2, 0, levels=(target, 0), level=2, cosine_threshold=0.9)
    canvas = g.add(
        "TFLevelCanvas", 2, 6, title="paint over this",
        pipeline=(pipe, 0), levels=(target, 0), level=2, view="latent PCA",
        draw_grid=True, size=512, regions=(regions, 0),
    )
    painter = g.add("Painter", 3, 0, image=(canvas, 0))
    tgt_tokens = g.add(
        "TFTokensFromMask", 4, 0,
        mask=(painter, 1), levels=(target, 0), coverage=0.35,
        regions=(regions, 0), region_overlap=0.3,
    )
    src_tokens = g.add("TFTokensFromCoords", 4, 6, title="source tokens",
                       coords="6,6:9 7,6:9", levels=(source, 0))
    edit = g.add(
        "TFFeatureEdit", 5, 0,
        levels=(target, 0), level=2, target_tokens=(tgt_tokens, 0), source_tokens=(src_tokens, 0),
        source_mode="region mean", strength=1.0, source_level=2, source_levels=(source, 0),
    )
    resume = g.add("TFResumeFromLevel", 6, 0, pipeline=(pipe, 0), levels=(edit, 0),
                   follow_edit=True, class_id=-1, seed=592)
    after = g.add("TFDecode", 7, 0, pipeline=(pipe, 0), levels=(resume, 0), which="all levels")
    g.add("PreviewImage", 8, 0, title="edited", images=(after, 0))
    check = g.add("TFTokensPreview", 5, 8, title="what the paint selected", tokens=(tgt_tokens, 0))
    g.add("PreviewImage", 6, 8, images=(check, 0))
    return g


def shape_edit() -> Graph:
    g = Graph(
        "04-shape-edit",
        "Move a boundary instead of a feature: hand tokens from one region to a neighbour, so the "
        "region's extent changes and its feature content does not.",
    )
    pipe = g.add("TFLoadPipeline", 0, 0)
    levels = g.add("TFGenerate", 1, 0, pipeline=(pipe, 0), class_id=213, seed=592)
    regions = g.add("TFRegionMap", 2, 0, levels=(levels, 0), level=2, cosine_threshold=0.9)
    boundary = g.add("TFTokensFromCoords", 2, 6, title="tokens to hand over",
                     coords="7,7 8,7", levels=(levels, 0))
    receiving = g.add("TFTokensFromCoords", 2, 10, title="one token in the receiving region",
                      coords="0,0", levels=(levels, 0))
    edit = g.add(
        "TFShapeEdit", 3, 0,
        levels=(levels, 0), level=2, regions=(regions, 0),
        boundary_tokens=(boundary, 0), receiving_tokens=(receiving, 0), strength=1.0,
    )
    resume = g.add("TFResumeFromLevel", 4, 0, pipeline=(pipe, 0), levels=(edit, 0),
                   follow_edit=True, class_id=-1, seed=592)
    after = g.add("TFDecode", 5, 0, pipeline=(pipe, 0), levels=(resume, 0), which="all levels")
    g.add("PreviewImage", 3, 8, title="regions", images=(regions, 1))
    g.add("PreviewImage", 6, 0, title="after the shape edit", images=(after, 0))
    return g


BUILDERS = [generate_and_decode, feature_edit_coords, feature_edit_painter, shape_edit]


def main() -> int:
    import asyncio

    # comfy.model_management calls torch.cuda.current_device() at import unless
    # --cpu is set, so without this the script only runs on a GPU node -- absurd
    # for something that reads schemas. Argv is only consulted after main.py
    # opts in, hence enable_args_parsing.
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import nodes as comfy_nodes

    # Loads every core node plus this extension through its real entrypoint, so
    # the schemas here are exactly the ones the running server will publish.
    asyncio.run(comfy_nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    missing = [n for n in ("TFLoadPipeline", "Painter") if n not in comfy_nodes.NODE_CLASS_MAPPINGS]
    if missing:
        raise SystemExit(f"nodes did not register: {missing} -- check the custom_nodes symlink")

    for build in BUILDERS:
        build().write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
