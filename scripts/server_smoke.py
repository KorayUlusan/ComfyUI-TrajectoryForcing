#!/usr/bin/env python3
"""Run the example workflows through a real ComfyUI server.

scripts/gpu_smoke.py calls the nodes directly, which leaves two things untested:
whether ComfyUI's execution engine is happy with the custom socket types (it
validates every link against /object_info before running anything), and whether
the generated workflows in workflows/api/ are actually executable rather than
merely well-formed JSON. Both only fail once a server is running, so this starts
one.

Exit criteria, fixed before the run:

  1. startup   -- the server answers /object_info, and every node this extension
                  declares is in it with the socket types it declared.
  2. execute   -- 01, 02, 04 and 05 each run to completion with no node errors,
                  and produce preview images. 05 is the sweep, so this is also
                  the only check that the loop survives ComfyUI's own execution
                  engine rather than a direct call -- it runs several re-samples
                  inside one node, which is not a shape any other node here has.
  3. edit      -- 02's edited output differs from its unedited control branch.
                  A workflow that runs but changes nothing would pass criterion
                  2 while the edit silently did nothing.
  4. widget    -- the clickable token grid is registered and served. It is the
                  only thing in this repo no test can exercise -- there is no
                  browser here -- so this checks the half that is checkable:
                  that ComfyUI found WEB_DIRECTORY, lists the script among its
                  extensions, and serves the right file. A wrong path or a
                  renamed directory fails here instead of silently leaving
                  everyone typing coordinates.
  5. sweep     -- 05's report names every arm it was asked for and reports a
                  spread, so the table reached the client rather than only the
                  images. A sweep that silently ran one arm would pass 2.
  6. blocked   -- 03 (the Painter one) stops at TF Tokens From Mask with no
                  error dialog at all, on *two* consecutive runs -- the bug this
                  replaced raised on the first run and went silent on the
                  second, so one run cannot tell the two apart. It must still
                  produce the canvas to paint on, and the reason must reach the
                  user as that node's own output text.

Usage (inside a GPU allocation):  python scripts/server_smoke.py [PORT]
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

EXT_ROOT = Path(__file__).resolve().parent.parent


def comfy_root() -> Path:
    env = os.environ.get("COMFY_DIR", "").strip()
    for path in ([Path(env)] if env else []) + [EXT_ROOT.parent.parent / "ComfyUI", Path.home() / "ComfyUI"]:
        if (path / "main.py").is_file():
            return path
    raise SystemExit("Could not find ComfyUI; set COMFY_DIR.")


COMFY = comfy_root()
OUT = EXT_ROOT / "outputs" / "server_smoke"
STARTUP_TIMEOUT = 900     # a cold TF Load Pipeline compiles XLA on first use
EXECUTE_TIMEOUT = 900

EXPECTED_NODES = {
    "TFLoadPipeline", "TFImageNetClass", "TFGenerate", "TFDecode", "TFLatentPreview",
    "TFLevelsInfo", "TFLevelCanvas", "TFRegionMap", "TFTokensFromMask",
    "TFTokensFromCoords", "TFTokensCombine", "TFTokensPreview", "TFFeatureEdit",
    "TFShapeEdit", "TFResumeFromLevel", "TFSaveLevels", "TFLoadLevels",
    "TFCompareLevels", "TFSweep", "TFSaveReport",
}

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def start_server(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "comfyui.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", "main.py", "--listen", "127.0.0.1", "--port", str(port),
         "--disable-auto-launch"],
        cwd=COMFY, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"started ComfyUI pid={proc.pid}, log at {OUT / 'comfyui.log'}", flush=True)
    return proc


def wait_for(url: str, proc: subprocess.Popen, timeout: int) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"server exited early with code {proc.returncode}", flush=True)
            return None
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def run_workflow(base: str, name: str, client_id: str) -> tuple[bool, dict, str]:
    payload = json.loads((EXT_ROOT / "workflows" / "api" / f"{name}.json").read_text())
    response = requests.post(f"{base}/prompt", json={"prompt": payload, "client_id": client_id})
    if not response.ok:
        return False, {}, f"validation rejected it: {response.text[:600]}"
    prompt_id = response.json()["prompt_id"]

    deadline = time.time() + EXECUTE_TIMEOUT
    while time.time() < deadline:
        history = requests.get(f"{base}/history/{prompt_id}", timeout=10).json()
        entry = history.get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None:
            status = entry["status"]
            if status.get("status_str") != "success":
                messages = [m for m in status.get("messages", []) if "error" in str(m[0])]
                return False, entry, f"execution failed: {json.dumps(messages)[:600]}"
            return True, entry, ""
        time.sleep(3)
    return False, {}, f"still running after {EXECUTE_TIMEOUT}s"


async def run_and_watch(base: str, name: str, client_id: str) -> list[dict]:
    """Run a workflow with a websocket open, and return everything the UI was told.

    A blocked node reports itself over the websocket only -- `execution_block_cb`
    calls `server.send_sync("execution_error", ...)` and the prompt still ends
    "success", so neither /history nor the server log records it. Polling
    /history therefore cannot tell an instructive stop from nothing happening at
    all, which is exactly the distinction this criterion is about.
    """
    import aiohttp

    payload = json.loads((EXT_ROOT / "workflows" / "api" / f"{name}.json").read_text())
    events: list[dict] = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"{base}/ws?clientId={client_id}") as ws:
            await session.post(f"{base}/prompt", json={"prompt": payload, "client_id": client_id})
            deadline = time.time() + EXECUTE_TIMEOUT
            while time.time() < deadline:
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=30)
                except TimeoutError:
                    break
                if message.type is not aiohttp.WSMsgType.TEXT:
                    continue  # binary frames are preview images
                event = json.loads(message.data)
                events.append(event)
                if event.get("type") in ("execution_success", "execution_interrupted"):
                    break
    return events


def fetch_images(base: str, entry: dict, stem: str) -> int:
    """Pull every preview image the run produced, so the result can be eyeballed."""
    OUT.mkdir(parents=True, exist_ok=True)
    # Clear this workflow's previous files first: frame counts change between
    # runs -- four decoded levels became one stitched sheet -- and last run's
    # leftovers sitting beside this run's read as part of it.
    for stale in OUT.glob(f"{stem}-*.png"):
        stale.unlink()
    saved = 0
    for node_id, output in entry.get("outputs", {}).items():
        for i, image in enumerate(output.get("images", [])):
            data = requests.get(f"{base}/view", params={
                "filename": image["filename"], "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "temp"),
            }, timeout=30)
            if data.ok:
                (OUT / f"{stem}-node{node_id}-{i}.png").write_bytes(data.content)
                saved += 1
    return saved


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
    base = f"http://127.0.0.1:{port}"
    client_id = str(uuid.uuid4())
    proc = start_server(port)
    try:
        # --- 1. startup ------------------------------------------------------
        info = wait_for(f"{base}/object_info", proc, STARTUP_TIMEOUT)
        if info is None:
            check("startup", False, "server never answered /object_info")
            print((OUT / "comfyui.log").read_text()[-4000:], flush=True)
            return 1
        missing = EXPECTED_NODES - set(info)
        types_ok = (
            info.get("TFGenerate", {}).get("output") == ["TF_LEVELS"]
            and info.get("TFRegionMap", {}).get("output") == ["TF_REGIONS", "IMAGE", "INT", "INT"]
            and "TF_PIPELINE" in json.dumps(info.get("TFGenerate", {}).get("input", {}))
            # the pipeline socket must survive as an optional override, or a
            # trajectory from TF Load Levels can never be decoded
            and "pipeline" in info.get("TFDecode", {}).get("input", {}).get("optional", {})
        )
        check(
            "startup",
            not missing and types_ok,
            f"{len(EXPECTED_NODES)} nodes registered, custom socket types published"
            if not missing else f"missing {sorted(missing)}",
        )

        # --- widget: registered and served -----------------------------------
        served, body = "", ""
        try:
            listed = requests.get(f"{base}/extensions", timeout=30).json()
            served = next((u for u in listed if u.endswith("/tf_token_grid.js")), "")
            if served:
                body = requests.get(f"{base}{served}", timeout=30).text
        except Exception as error:  # noqa: BLE001 - reported, not raised
            served = f"<{type(error).__name__}: {error}>"
        check(
            "the token grid widget is registered and served",
            bool(served) and "TFTokensFromCoords" in body and "addDOMWidget" in body,
            f"{served} ({len(body)} bytes)" if body else f"not served: {served or 'absent from /extensions'}",
        )

        # --- 2 & 3. execute --------------------------------------------------
        entries = {}
        for name in ("01-generate-and-decode", "02-feature-edit-coords", "04-shape-edit",
                     "05-sweep-seeds"):
            started = time.perf_counter()
            ok, entry, detail = run_workflow(base, name, client_id)
            images = fetch_images(base, entry, name) if ok else 0
            entries[name] = entry
            check(
                f"execute {name}",
                ok and images > 0,
                detail or f"{images} images in {time.perf_counter() - started:.1f}s",
            )

        edited = entries.get("02-feature-edit-coords", {}).get("outputs", {})
        edited_files = {
            image["filename"]
            for output in edited.values() for image in output.get("images", [])
        }
        check(
            "edit changed something",
            len(edited_files) >= 2 and len({
                (OUT / p.name).read_bytes()
                for p in OUT.glob("02-feature-edit-coords-*.png")
            }) > 1,
            f"{len(edited_files)} distinct preview images from the edit workflow",
        )

        # --- 4. the sweep's table reaches the client --------------------------
        sweep_said = json.dumps(entries.get("05-sweep-seeds", {}).get("outputs", {}))
        arms_named = [s for s in ("592", "593", "594", "595") if s in sweep_said]
        check(
            "the sweep reports every arm and a spread",
            len(arms_named) == 4 and "spread across arms" in sweep_said,
            f"{len(arms_named)}/4 arms named in the node's own output text"
            if "spread across arms" in sweep_said else f"no table in {sweep_said[:400]}",
        )

        # --- 5. the painter workflow's first two runs ------------------------
        # Run it twice. The bug this replaced showed an error dialog on the
        # first run and nothing on the second, because the second found the node
        # cached -- so a single run cannot tell whether it is fixed.
        for attempt in (1, 2):
            events = asyncio.run(run_and_watch(base, "03-feature-edit-painter", client_id))
            errors = [e["data"] for e in events if e.get("type") == "execution_error"]
            check(
                f"03-feature-edit-painter, run {attempt}: stops without an error dialog",
                not errors,
                "blocked quietly at TF Tokens From Mask"
                if not errors else f"raised {json.dumps(errors)[:400]}",
            )

        _, entry, _ = run_workflow(base, "03-feature-edit-painter", client_id)
        said = json.dumps(entry.get("outputs", {}))
        check(
            "and it says why, in the node rather than a modal",
            "Nothing painted yet" in said,
            "TF Tokens From Mask carries the instruction as its own output text"
            if "Nothing painted yet" in said else f"no notice in {said[:400]}",
        )
        check(
            "and its first run still produced the canvas to paint on",
            fetch_images(base, entry, "03-feature-edit-painter") > 0,
            "TF Level Canvas previewed before the graph stopped",
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 60, flush=True)
    print(f"{len(_results) - len(failed)}/{len(_results)} criteria passed; images in {OUT}", flush=True)
    if failed:
        print("FAILED: " + ", ".join(failed), flush=True)
        print("\n--- last 3000 chars of the server log ---", flush=True)
        print((OUT / "comfyui.log").read_text()[-3000:], flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
