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
# Slack around a group's members. Columns are COLUMN apart and nodes are
# COLUMN-60 wide, so anything above 30 makes adjacent groups share an edge --
# which reads as one box rather than two. 22 leaves a visible 16px gutter.
GROUP_PAD = 22
GROUP_TITLE_SPACE = 42  # room for the group's own title bar above them

# Group colours, in the order groups are declared. These are LiteGraph's own
# palette values, so they match what the "Colour" menu offers and stay legible
# in both the light and dark themes.
GROUP_COLOURS = ["#3f789e", "#88A", "#8A8", "#a1309b", "#b58b2a", "#3f789e"]

# Minimum height for nodes that display something in their body once the graph
# has run. A node sized for its empty state overflows its group and collides
# with its neighbour the moment an image arrives, and the first time anyone sees
# the workflow is usually *after* pressing Run.
BODY_HEIGHT = {
    "PreviewImage": 430,      # a 512px image scaled into a 320-wide node, plus chrome
    "Painter": 620,           # its canvas widget plus the brush controls beneath
    "TFLevelCanvas": 450,     # publishes a preview so Painter has a backdrop
    "TFDecode": 150,          # ui.PreviewText: a couple of lines
    "TFLatentPreview": 150,
    "TFCompareLevels": 260,   # a per-level table
    "TFTokensFromMask": 220,  # the "nothing painted yet" instruction
}
ROW_GAP = 26                  # vertical breathing room between stacked nodes


class Graph:
    """Collects nodes and links, then serialises to both workflow formats."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self.groups: list[dict] = []
        self._group_specs: list[tuple[str, list[int], str]] = []
        # The API payload is built alongside the LiteGraph one rather than
        # derived from it afterwards: reconstructing which slot a value came
        # from means re-deriving the widget/socket split a second time, and the
        # two derivations drifting apart is exactly the bug this generator is
        # meant to prevent.
        self.api: dict[str, dict] = {}

    def note(self, text: str, column: float, row: float, title: str = "",
             width: int = 340, height: int = 260) -> int:
        """A MarkdownNote: explanatory text on the canvas.

        Frontend-only (`isVirtualNode`), so it is deliberately absent from the
        API payload -- posting one to /prompt would be rejected, since the
        server has no node class by that name.
        """
        node_id = len(self.nodes) + 1
        self.nodes.append({
            "id": node_id,
            "type": "MarkdownNote",
            "pos": [80 + int(column * COLUMN), 80 + int(row * ROW)],
            "size": [width, height],
            "flags": {},
            "order": node_id - 1,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {},
            "widgets_values": [text.strip() + "\n"],
            "color": "#432",
            "bgcolor": "#653",
            **({"title": title} if title else {}),
        })
        return node_id

    def group(self, title: str, members: list[int], colour: str = "") -> None:
        """Box and label a set of nodes.

        The box is computed from where the members end up, not written down: a
        hand-typed bounding box drifts off its nodes the moment one moves, and a
        group that no longer contains its nodes leaves them behind when dragged.
        Resolved in `_layout`, after nodes have been spaced apart, since that is
        the first point at which the final positions are known.
        """
        self._group_specs.append((title, list(members),
                                  colour or GROUP_COLOURS[len(self._group_specs) % len(GROUP_COLOURS)]))

    def _stack_columns(self, group_of: dict[int, int]) -> None:
        """Push nodes down a column until none overlaps the one above it."""
        # Two nodes from different groups need room for both group borders and
        # the lower one's title bar between them, or the boxes overlap even
        # though the nodes do not -- and an overlapping group picks up its
        # neighbour's nodes when dragged.
        across_groups = 2 * GROUP_PAD + GROUP_TITLE_SPACE + ROW_GAP
        for column in {n["pos"][0] for n in self.nodes}:
            in_column = sorted((n for n in self.nodes if n["pos"][0] == column),
                               key=lambda n: n["pos"][1])
            previous = None
            for node in in_column:
                if previous is not None:
                    gap = (across_groups
                           if group_of.get(node["id"]) != group_of.get(previous["id"])
                           else ROW_GAP)
                    node["pos"][1] = max(
                        node["pos"][1], previous["pos"][1] + previous["size"][1] + gap)
                previous = node

    def _box(self, members: list[int]) -> list[int]:
        boxes = [self.nodes[i - 1] for i in members]
        left = min(n["pos"][0] for n in boxes) - GROUP_PAD
        top = min(n["pos"][1] for n in boxes) - GROUP_PAD - GROUP_TITLE_SPACE
        right = max(n["pos"][0] + n["size"][0] for n in boxes) + GROUP_PAD
        bottom = max(n["pos"][1] + n["size"][1] for n in boxes) + GROUP_PAD
        return [left, top, right - left, bottom - top]

    def _layout(self) -> None:
        """Space the nodes out, then box the groups.

        Row numbers in the builders are an ordering hint. Honouring them
        literally means predicting every node height by hand, and one node
        growing an image preview then silently overlaps the next -- which is
        what happened. A node keeps its requested row when there is room and is
        pushed down when there is not.

        Column stacking alone is not enough for the groups: a group spanning two
        columns extends as far as its lowest member in *either*, so two groups
        can interleave while no two nodes overlap. The loop settles that by
        pushing a whole group down and re-stacking, which can in turn move
        something else -- hence iterating rather than a single pass.
        """
        group_of = {}
        for index, (_, members, _) in enumerate(self._group_specs):
            for member in members:
                group_of[member] = index

        for _ in range(len(self._group_specs) + 2):
            self._stack_columns(group_of)
            boxes = [self._box(members) for _, members, _ in self._group_specs]
            moved = False
            for lower in range(len(boxes)):
                for upper in range(lower):
                    ax, ay, aw, ah = boxes[upper]
                    bx, by, bw, bh = boxes[lower]
                    overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
                    overlap_y = min(ay + ah, by + bh) - max(ay, by)
                    if overlap_x > 0 and overlap_y > 0:
                        shift = (ay + ah) - by + ROW_GAP
                        for member in self._group_specs[lower][1]:
                            self.nodes[member - 1]["pos"][1] += shift
                        boxes[lower] = self._box(self._group_specs[lower][1])
                        moved = True
            if not moved:
                break

        self.groups = [
            {
                "id": index + 1,
                "title": title,
                "bounding": self._box(members),
                "color": colour,
                "font_size": 24,
                "flags": {},
            }
            for index, (title, members, colour) in enumerate(self._group_specs)
        ]

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
            "size": [COLUMN - 60,
                     max(40 + 26 * (len(inputs) + len(widgets)),
                         BODY_HEIGHT.get(node_type, 0))],
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
        self._layout()
        return {
            "id": f"trajectory-forcing-{self.name}",
            "revision": 0,
            "last_node_id": len(self.nodes),
            "last_link_id": len(self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
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
#
# Each is laid out left to right in the order it executes, boxed into numbered
# groups, and opens with a MarkdownNote explaining what you are looking at --
# the graph is the documentation for someone who has just double-clicked a file
# they were sent, and a wall of unlabelled nodes is not self-explanatory however
# good the node tooltips are.
# ---------------------------------------------------------------------------
def generate_and_decode() -> Graph:
    g = Graph(
        "01-generate-and-decode",
        "The vertical slice: sample one coarse-to-fine trajectory and look at every level, "
        "as decoded RGB and as the raw PCA token grid.",
    )
    g.note(
        """
# 1. Generate and look

**Start here.** Press **Run** and wait — the first run loads the model and takes
1–2 minutes. Later runs take about a second.

Trajectory Forcing does not paint an image in one go. It builds one in four
passes, coarse to fine:

| level | what it decides |
|---|---|
| 0 | object vs. background |
| 1 | parts |
| 2 | subparts |
| 3 | the finest detail |

Every level can be decoded to a picture, which is what the two preview strips on
the right show you. That is the whole point of the method: the intermediate
steps are things you can *look at*, and in the other workflows, edit.

**Try:** change the class in *TF ImageNet Class*, or the seed in *TF Generate*.
""",
        column=-1.15, row=0, title="README — read me first", width=400, height=560,
    )

    pipe = g.add("TFLoadPipeline", 0, 0)
    cls = g.add("TFImageNetClass", 0, 5, class_name=imagenet_option(213))
    g.group("1 · Load the model", [pipe, cls])

    gen = g.add("TFGenerate", 1, 0, pipeline=(pipe, 0), class_id=(cls, 0), seed=592)
    g.group("2 · Sample a trajectory", [gen])

    dec = g.add("TFDecode", 2, 0, levels=(gen, 0), which="all levels")
    pca = g.add("TFLatentPreview", 2, 6, levels=(gen, 0), which="all levels")
    info = g.add("TFLevelsInfo", 2, 12, levels=(gen, 0))
    rgb_view = g.add("PreviewImage", 3, 0, title="Decoded RGB — one per level", images=(dec, 0))
    pca_view = g.add("PreviewImage", 3, 6, title="Latent PCA — one per level", images=(pca, 0))
    g.group("3 · Look at every level", [dec, pca, info, rgb_view, pca_view])
    return g


def feature_edit_coords() -> Graph:
    g = Graph(
        "02-feature-edit-coords",
        "Cross-image feature edit driven by typed token coordinates: take the mean feature of a "
        "region of one trajectory, write it into a region of another, and re-sample the finer "
        "levels. Runs as-is with no painting, so it is the reproducible version.",
    )
    g.note(
        """
# 2. Edit a region, keep the rest

**Press Run — this one works as-is, no painting needed.**

Two images are generated: a **target** (the one being edited) and a **source**
(where the new content comes from). The edit takes the average feature of a
patch of the source and writes it into a region of the target, at level 2.
Then every finer level is re-sampled from that edited canvas, so the model
makes the change *fit* rather than pasting a rectangle.

Levels below the edit are untouched — an edit only ever propagates upward.

**Advanced widgets show `(-1 = auto)`** — that means the node works the value
out for itself, and it prints which it chose. Leave them alone unless you want
something specific.

**Try, in the yellow group:**
- *target region*: change `7,7` to another `row,col` on the 16×16 grid. It snaps
  to the whole region containing that token. Every preview here is numbered
  along its top and left edge, so you can read a coordinate off rather than
  count cells.
- *source tokens*: `6,6:9` means row 6, columns 6 through 9.
- *TF Feature Edit → level*: 0–3. Lower = a bigger, more semantic change.
- *strength*: below 1.0 blends instead of replacing.

The bottom-right preview shows exactly which tokens were selected, drawn on the
level you edited. Check it if a result surprises you.

*TF Compare Levels* at the far bottom answers "did it actually do anything":
tokens changed per level, and a heatmap of where. An edit at level 2 should
leave levels 0 and 1 completely untouched.

The edit's `level` is wired from *TF Region Map* rather than typed twice — a
selection snapped to one level's regions is refused at another, because the
token grid is the same size at every level and it would otherwise land on the
wrong part of the image without complaining.
""",
        column=-1.15, row=0, title="README — read me first", width=400, height=580,
    )

    pipe = g.add("TFLoadPipeline", 0, 0)
    target = g.add("TFGenerate", 1, 0, title="target — the image being edited",
                   pipeline=(pipe, 0), class_id=213, seed=592)
    source = g.add("TFGenerate", 1, 6, title="source — where the new feature comes from",
                   pipeline=(pipe, 0), class_id=207, seed=592)
    g.group("1 · Load, and generate two images", [pipe, target, source])

    regions = g.add("TFRegionMap", 2, 0, levels=(target, 0), level=2, cosine_threshold=0.9)
    tgt_tokens = g.add(
        "TFTokensFromCoords", 2, 6, title="target region — WHERE the edit lands",
        coords="7,7", levels=(target, 0), regions=(regions, 0),
    )
    src_tokens = g.add(
        "TFTokensFromCoords", 2, 11, title="source tokens — WHAT gets written there",
        coords="6,6:9 7,6:9", levels=(source, 0),
    )
    g.group("2 · Choose what to edit", [regions, tgt_tokens, src_tokens], colour="#b58b2a")

    # `level` comes from the region map rather than being typed again: the two
    # have to agree, and a selection snapped to another level's regions is now
    # refused rather than silently landing on the wrong part of the image.
    edit = g.add(
        "TFFeatureEdit", 3, 0,
        levels=(target, 0), level=(regions, 3), target_tokens=(tgt_tokens, 0),
        source_tokens=(src_tokens, 0), source_mode="region mean", strength=1.0, source_levels=(source, 0),
    )
    resume = g.add(
        "TFResumeFromLevel", 4, 0,
        levels=(edit, 0), class_id=-1, seed=592,
    )
    g.group("3 · Apply it, then re-sample the finer levels", [edit, resume], colour="#a1309b")

    after = g.add("TFDecode", 5, 0, title="after", levels=(resume, 0), which="all levels")
    before = g.add("TFDecode", 5, 6, title="before", levels=(target, 0),
                   which="final level only")
    after_view = g.add("PreviewImage", 6, 0, title="EDITED — all four levels", images=(after, 0))
    before_view = g.add("PreviewImage", 6, 6, title="UNEDITED — for comparison",
                        images=(before, 0))
    canvas = g.add("TFLevelCanvas", 5, 10, title="the level you edited, with the selection on it",
                   levels=(target, 0), level=(regions, 3),
                   view="latent PCA", regions=(regions, 0), highlight=(tgt_tokens, 0))
    canvas_view = g.add("PreviewImage", 6, 10, title="what was selected", images=(canvas, 0))
    g.group("4 · Compare", [after, before, after_view, before_view, canvas, canvas_view],
            colour="#8A8")

    compare = g.add("TFCompareLevels", 5, 21, before=(target, 0), after=(resume, 0), size=512)
    compare_view = g.add("PreviewImage", 6, 21, title="where it changed, per level",
                         images=(compare, 1))
    g.group("5 · Measure it", [compare, compare_view], colour="#88A")
    return g


def feature_edit_painter() -> Graph:
    g = Graph(
        "03-feature-edit-painter",
        "The same edit, but the target region is painted. Run TF Level Canvas once, paint on the "
        "Painter node over the token grid it produces, then run again -- the mask is snapped to "
        "whole cosine regions, so a rough stroke still selects a clean part.",
    )
    g.note(
        """
# 3. Paint the region instead of typing it

Same edit as workflow 2, but you choose the region with a brush.

**This one takes two runs. That is normal, not an error.**

1. **Press Run.** It stops partway with a note in *TF Tokens From Mask* saying
   nothing is painted yet. The canvas appears **inside the Painter node**, ready
   to paint over.
2. **Paint** on the Painter node, over the area you want to change.
3. **Press Run again.** Now it finishes.

> **If the Painter says "Node 2.0 only"** it cannot be painted on: that widget
> only exists in ComfyUI's new node rendering. Turn it on in **Settings (gear,
> bottom left) → search "Node 2.0" → enable**, then reload the page.
> *TF Level Canvas* also says so if it detects the setting is off.
> Workflow 02 types coordinates instead and needs none of this.

The grid drawn on the canvas is the 16×16 token grid — one cell is one token,
the smallest thing the edit can address. The yellow lines are region boundaries.
Your brush stroke is snapped to whole regions, so it does not have to be neat;
covering most of a region is enough.

If it still stops after painting, read the note in *TF Tokens From Mask* — it
says whether you painted too thinly (*coverage*) or across too many regions
(*region_overlap*).
""",
        column=-1.15, row=0, title="README — read me first", width=400, height=580,
    )

    pipe = g.add("TFLoadPipeline", 0, 0)
    target = g.add("TFGenerate", 1, 0, title="target — the image being edited",
                   pipeline=(pipe, 0), class_id=213, seed=592)
    source = g.add("TFGenerate", 1, 6, title="source — where the new feature comes from",
                   pipeline=(pipe, 0), class_id=207, seed=592)
    g.group("1 · Load, and generate two images", [pipe, target, source])

    regions = g.add("TFRegionMap", 2, 0, levels=(target, 0), level=2, cosine_threshold=0.9)
    canvas = g.add(
        "TFLevelCanvas", 2, 6, title="the canvas — this is what you paint on",
        levels=(target, 0), level=(regions, 3), view="latent PCA",
        draw_grid=True, size=512, regions=(regions, 0),
    )
    # Previewed on its own so the first run -- which stops at TF Tokens From
    # Mask, there being nothing painted yet -- still visibly produces the thing
    # you are meant to paint on, instead of looking like it did nothing.
    canvas_view = g.add("PreviewImage", 2, 12, title="run once to see this, then paint",
                        images=(canvas, 0))
    painter = g.add("Painter", 3, 0, title="PAINT HERE, then press Run again",
                    image=(canvas, 0))
    tgt_tokens = g.add(
        "TFTokensFromMask", 4, 0, title="your stroke, snapped to whole regions",
        mask=(painter, 1), levels=(target, 0), coverage=0.35,
        regions=(regions, 0), region_overlap=0.3,
    )
    check = g.add("TFTokensPreview", 4, 7, title="what your paint actually selected",
                  tokens=(tgt_tokens, 0))
    check_view = g.add("PreviewImage", 4, 11, images=(check, 0))
    g.group("2 · Paint the region  ← the two-run step",
            [regions, canvas, canvas_view, painter, tgt_tokens, check, check_view],
            colour="#b58b2a")

    src_tokens = g.add("TFTokensFromCoords", 5, 8, title="source tokens — WHAT gets written",
                       coords="6,6:9 7,6:9", levels=(source, 0))
    edit = g.add(
        "TFFeatureEdit", 5, 0,
        levels=(target, 0), level=(regions, 3), target_tokens=(tgt_tokens, 0),
        source_tokens=(src_tokens, 0), source_mode="region mean", strength=1.0, source_levels=(source, 0),
    )
    resume = g.add("TFResumeFromLevel", 6, 0, levels=(edit, 0),
                   class_id=-1, seed=592)
    g.group("3 · Apply it, then re-sample the finer levels", [src_tokens, edit, resume],
            colour="#a1309b")

    after = g.add("TFDecode", 7, 0, levels=(resume, 0), which="all levels")
    after_view = g.add("PreviewImage", 8, 0, title="EDITED — all four levels", images=(after, 0))
    g.group("4 · Result", [after, after_view], colour="#8A8")
    return g


def shape_edit() -> Graph:
    g = Graph(
        "04-shape-edit",
        "Move a boundary instead of a feature: hand tokens from one region to a neighbour, so the "
        "region's extent changes and its feature content does not.",
    )
    g.note(
        """
# 4. Change a region's shape, not its content

**Press Run — this one works as-is.**

The other edits change *what* a region looks like. This one changes *where it
ends*. Tokens are handed from one region to a neighbour: they take on the
receiving region's average feature, so the boundary moves and nothing new is
invented.

Two selections, both on the 16×16 token grid:

- **tokens to hand over** — the ones that change sides.
- **one token in the receiving region** — anywhere inside the region that is
  taking them. The whole region it names supplies the feature.

They must name **different** regions, or the node stops and says so.

**Look at the region map preview first** to see where the boundaries actually
are, then pick coordinates on either side of one.
""",
        column=-1.15, row=0, title="README — read me first", width=400, height=520,
    )

    pipe = g.add("TFLoadPipeline", 0, 0)
    levels = g.add("TFGenerate", 1, 0, pipeline=(pipe, 0), class_id=213, seed=592)
    g.group("1 · Load and generate", [pipe, levels])

    regions = g.add("TFRegionMap", 2, 0, levels=(levels, 0), level=2, cosine_threshold=0.9)
    region_view = g.add("PreviewImage", 2, 6, title="the regions — pick coordinates from this",
                        images=(regions, 1))
    boundary = g.add("TFTokensFromCoords", 3, 0, title="tokens to hand over",
                     coords="7,7 8,7", levels=(levels, 0))
    receiving = g.add("TFTokensFromCoords", 3, 5, title="one token in the receiving region",
                      coords="0,0", levels=(levels, 0))
    g.group("2 · Find a boundary, name both sides",
            [regions, region_view, boundary, receiving], colour="#b58b2a")

    edit = g.add(
        "TFShapeEdit", 4, 0,
        levels=(levels, 0), level=(regions, 3), regions=(regions, 0),
        boundary_tokens=(boundary, 0), receiving_tokens=(receiving, 0), strength=1.0,
    )
    resume = g.add("TFResumeFromLevel", 5, 0, levels=(edit, 0),
                   class_id=-1, seed=592)
    g.group("3 · Move the boundary, then re-sample", [edit, resume], colour="#a1309b")

    after = g.add("TFDecode", 6, 0, levels=(resume, 0), which="all levels")
    after_view = g.add("PreviewImage", 7, 0, title="after the shape edit", images=(after, 0))
    g.group("4 · Result", [after, after_view], colour="#8A8")
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
