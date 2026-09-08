#!/usr/bin/env python3
"""Strip the example workflows down to filming versions.

The examples in example_workflows/ are built to TEACH: a screenful of
MarkdownNote, a comparison node, several previews. That is right for someone
opening the repo and wrong for a screen recording, where the canvas has to be
zoomed far enough in to read node titles and everything on screen competes with
the one action being filmed.

These are the same graphs with the teaching furniture removed and the widget
values pre-set to the shot, so the recording is: open, zoom, do the one thing,
Queue. See _marketing/teaser-comfyui-tf/RECORDING_NODES.md.

DERIVED, NOT HAND-WRITTEN. LiteGraph stores widget values POSITIONALLY, so a
hand-written workflow silently loads plausible wrong numbers the moment a widget
is added or reordered upstream -- which is the whole reason make_workflows.py
exists. This reads the generated files and edits them by node type, so it
inherits that protection rather than working around it. Re-run it after
make_workflows.py, never instead of it.

    python3 scripts/make_recording_workflows.py
"""
from __future__ import annotations

import json
from pathlib import Path

EXT = Path(__file__).resolve().parent.parent
SRC = EXT / "example_workflows"
OUT = EXT / "example_workflows" / "recording"

# Node types that are furniture for a reader and clutter for a camera.
DROP_TYPES = {"MarkdownNote", "TFCompareLevels", "TFLevelCanvas", "Note"}


def prune(doc: dict, drop_ids: set[int]) -> dict:
    """Remove nodes and every link that touches them, leaving a valid graph.

    Dropping a node without dropping its links leaves LiteGraph with edges
    pointing at nothing; it loads, then throws on execute, which is the worst
    possible moment to find out -- mid-take.
    """
    doc["nodes"] = [n for n in doc["nodes"] if n["id"] not in drop_ids]
    kept = {n["id"] for n in doc["nodes"]}

    doc["links"] = [
        lk for lk in doc.get("links", [])
        if lk[1] in kept and lk[3] in kept
    ]
    live = {lk[0] for lk in doc["links"]}

    for n in doc["nodes"]:
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None and inp["link"] not in live:
                inp["link"] = None
        for out in n.get("outputs", []) or []:
            if out.get("links"):
                out["links"] = [x for x in out["links"] if x in live]

    # Groups whose members are gone would draw as empty boxes.
    doc["groups"] = [
        g for g in doc.get("groups", [])
        if any(inside(n, g) for n in doc["nodes"])
    ]
    return doc


def drop_orphan_previews(doc: dict) -> dict:
    """Remove PreviewImage nodes whose source was pruned.

    Pruning a TFDecode or TFLevelCanvas leaves its preview behind with a dead
    input. It renders as an empty grey box that does nothing on Run -- which on
    camera looks exactly like a node that failed.
    """
    orphans = {
        n["id"] for n in doc["nodes"]
        if n["type"] == "PreviewImage"
        and not any(i.get("link") is not None for i in (n.get("inputs") or []))
    }
    return prune(doc, orphans) if orphans else doc


def validate(doc: dict, name: str) -> None:
    """Every link must join two nodes that still exist, and every input that
    claims a link must name one that is in the link table. A workflow that
    loads and then throws on Run is the worst failure mode here."""
    ids = {n["id"] for n in doc["nodes"]}
    links = {lk[0] for lk in doc["links"]}
    for lk in doc["links"]:
        assert lk[1] in ids and lk[3] in ids, f"{name}: link {lk[0]} dangles"
    for n in doc["nodes"]:
        for i in n.get("inputs") or []:
            assert i.get("link") is None or i["link"] in links, \
                f"{name}: {n['type']} input '{i.get('name')}' -> missing link {i['link']}"
        for o in n.get("outputs") or []:
            for x in o.get("links") or []:
                assert x in links, f"{name}: {n['type']} output -> missing link {x}"


def inside(node: dict, group: dict) -> bool:
    b = group.get("bounding") or [0, 0, 0, 0]
    x, y = node.get("pos", [0, 0])[:2]
    return b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]


def set_widget(doc: dict, node_type: str, index: int, value, nth: int = 0) -> None:
    """Set one positional widget on the nth node of a type."""
    hits = [n for n in doc["nodes"] if n["type"] == node_type]
    if len(hits) <= nth:
        raise SystemExit(f"no {node_type}[{nth}] in workflow")
    w = hits[nth].setdefault("widgets_values", [])
    while len(w) <= index:
        w.append(None)
    w[index] = value


def retitle(doc: dict, node_type: str, title: str, nth: int = 0) -> None:
    hits = [n for n in doc["nodes"] if n["type"] == node_type]
    if len(hits) > nth:
        hits[nth]["title"] = title


def relayout(doc: dict, per_row: int = 3, dx: int = 430, dy: int = 300) -> dict:
    """Re-pack what is left into a compact block.

    After pruning, the survivors keep the coordinates they had in a much larger
    graph, so the canvas is mostly empty space and "fit to view" zooms out far
    enough that no node title is readable. That is precisely the failure this
    file exists to prevent.
    """
    order = sorted(doc["nodes"], key=lambda n: (n.get("order", 0)))
    for i, n in enumerate(order):
        n["pos"] = [80 + (i % per_row) * dx, 80 + (i // per_row) * dy]
    doc["groups"] = []  # positions no longer match; an empty canvas films better
    return doc


def build_feature_edit() -> dict:
    doc = json.loads((SRC / "02-feature-edit-coords.json").read_text())
    drop = {n["id"] for n in doc["nodes"] if n["type"] in DROP_TYPES}
    # Keep ONE preview of the region map and ONE of the result. The all-levels
    # contact sheet is the teaching view; the final image is the payoff.
    all_levels = [n for n in doc["nodes"]
                  if n["type"] == "TFDecode" and (n.get("widgets_values") or [None])[0] == "all levels"]
    drop |= {n["id"] for n in all_levels}
    doc = drop_orphan_previews(prune(doc, drop))

    # The shot: type the coords, drag strength, Queue.
    set_widget(doc, "TFTokensFromCoords", 0, "7,7", nth=0)        # target region
    set_widget(doc, "TFTokensFromCoords", 0, "6,6:9 7,6:9", nth=1)  # source tokens
    set_widget(doc, "TFFeatureEdit", 2, 1.0)                       # strength
    retitle(doc, "TFTokensFromCoords", "TYPE THE TARGET REGION  ->  7,7", nth=0)
    retitle(doc, "TFTokensFromCoords", "TYPE THE SOURCE TOKENS  ->  6,6:9 7,6:9", nth=1)
    retitle(doc, "TFFeatureEdit", "DRAG strength  0 -> 1.0")
    return relayout(doc)


def build_shape_edit() -> dict:
    doc = json.loads((SRC / "04-shape-edit.json").read_text())
    drop = {n["id"] for n in doc["nodes"] if n["type"] in DROP_TYPES}
    doc = drop_orphan_previews(prune(doc, drop))

    set_widget(doc, "TFTokensFromCoords", 0, "7,7 8,7", nth=0)   # tokens handed over
    set_widget(doc, "TFTokensFromCoords", 0, "0,0", nth=1)       # receiving region
    set_widget(doc, "TFShapeEdit", 1, 1.0)                       # strength
    retitle(doc, "TFTokensFromCoords", "TYPE THE TOKENS TO MOVE  ->  7,7 8,7", nth=0)
    retitle(doc, "TFTokensFromCoords", "TYPE THE RECEIVING REGION  ->  0,0", nth=1)
    retitle(doc, "TFShapeEdit", "DRAG strength  0 -> 1.0")
    return relayout(doc)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("rec-01-feature-edit", build_feature_edit),
        ("rec-02-shape-edit", build_shape_edit),
    ):
        doc = build()
        validate(doc, name)
        (OUT / f"{name}.json").write_text(json.dumps(doc, indent=1))
        types = [n["type"] for n in doc["nodes"]]
        print(f"{name}.json  {len(doc['nodes'])} nodes, {len(doc['links'])} links")
        for t in types:
            print(f"    {t}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
