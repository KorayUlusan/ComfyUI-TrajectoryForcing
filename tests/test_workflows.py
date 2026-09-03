"""Geometry of the generated workflow files.

These are checked rather than eyeballed because the failure mode is invisible
until the graph has *run*: a node sized for its empty state fits fine, then
grows an image preview and lands on top of its neighbour, and the group box
drawn around its empty size no longer contains it.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent / "workflows").glob("*.json"))


def boxes_overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return min(ax + aw, bx + bw) - max(ax, bx) > 0 and min(ay + ah, by + bh) - max(ay, by) > 0


def node_box(node) -> list[float]:
    return [node["pos"][0], node["pos"][1], node["size"][0], node["size"][1]]


@pytest.fixture(params=WORKFLOWS, ids=lambda p: p.stem)
def workflow(request):
    return json.loads(request.param.read_text())


@pytest.fixture
def api_workflow(request, workflow):
    name = request.node.callspec.params["workflow"].name
    return json.loads((WORKFLOWS[0].parent / "api" / name).read_text())


def test_there_are_workflows_to_check():
    assert len(WORKFLOWS) == 4


def test_no_two_nodes_overlap(workflow):
    for a, b in itertools.combinations(workflow["nodes"], 2):
        assert not boxes_overlap(node_box(a), node_box(b)), (
            f"{a.get('title', a['type'])} overlaps {b.get('title', b['type'])}")


def test_no_two_groups_overlap(workflow):
    # An overlapping group picks up its neighbour's nodes when dragged.
    for a, b in itertools.combinations(workflow["groups"], 2):
        assert not boxes_overlap(a["bounding"], b["bounding"]), (
            f"{a['title']!r} overlaps {b['title']!r}")


def test_every_group_contains_nodes(workflow):
    for group in workflow["groups"]:
        x, y, w, h = group["bounding"]
        inside = [n for n in workflow["nodes"]
                  if x <= n["pos"][0] and n["pos"][0] + n["size"][0] <= x + w
                  and y <= n["pos"][1] and n["pos"][1] + n["size"][1] <= y + h]
        assert inside, f"{group['title']!r} boxes nothing"


def test_image_bearing_nodes_are_sized_for_their_grown_state(workflow):
    # A PreviewImage is ~66px tall until it has an image and ~430 after. Sizing
    # for the empty state is what pushed content out of the groups.
    for node in workflow["nodes"]:
        if node["type"] == "PreviewImage":
            assert node["size"][1] >= 400, f"{node.get('title', node['id'])} is {node['size'][1]}px"


def test_notes_never_reach_the_api_payload(workflow, api_workflow):
    # MarkdownNote is a frontend-only virtual node; POSTing one is rejected.
    assert any(n["type"] == "MarkdownNote" for n in workflow["nodes"]), "each has a README note"
    assert not any(v["class_type"] == "MarkdownNote" for v in api_workflow.values())


def test_every_link_references_real_nodes_and_slots(workflow):
    by_id = {n["id"]: n for n in workflow["nodes"]}
    for link_id, src, src_slot, dst, dst_slot, io_type in workflow["links"]:
        assert src in by_id and dst in by_id, f"link {link_id} dangles"
        assert src_slot < len(by_id[src]["outputs"]), f"link {link_id} bad output slot"
        assert dst_slot < len(by_id[dst]["inputs"]), f"link {link_id} bad input slot"
        assert by_id[src]["outputs"][src_slot]["type"] == io_type
        assert by_id[dst]["inputs"][dst_slot]["type"] == io_type


def test_the_painter_image_input_is_slot_zero(workflow):
    # The Painter resolves its backdrop with getInputNode(0), so the image must
    # be the first socket or it paints on nothing.
    for node in workflow["nodes"]:
        if node["type"] == "Painter":
            assert node["inputs"][0]["name"] == "image"
