#!/usr/bin/env python3
"""Does the UI actually get told anything, through the ncat bridge?

`server_smoke.py` talks to ComfyUI on the compute node's own loopback, so it
proves the server is right and says nothing about the path a person actually
uses: browser -> login node -> `ncat` bridge -> compute node. That bridge is the
one component nothing tested, and it is where two reports landed --

    "while this is happening I see no feedback in the UI, no loading bar"
    "the image generates but is not visible in the UI output"

-- which are the same failure seen twice, because both travel on the websocket.
ComfyUI pushes `progress` events and the `executed` message naming a node's
output images over it; the graph runs regardless, so the server log and
/history look perfectly healthy while the browser is told nothing.

Run this from the login node. It launches a real session, drives workflow 01
through the bridge exactly as a browser would, and reports what arrived.

    python scripts/bridge_smoke.py

Exit criteria, fixed before the run:

  1. bridge     -- the login node's localhost:PORT answers /system_stats.
  2. websocket  -- a ws:// connection through the bridge stays open for the
                  whole run and reports the prompt finished.
  3. progress   -- at least one `progress` event arrives while TF Load Pipeline
                  is compiling. Without it a first run is a minute of nothing.
  4. images     -- an `executed` message carries the output images. This is the
                  message a preview node needs; /history having the images
                  proves only that the server made them.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parent.parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8188
BASE = f"http://127.0.0.1:{PORT}"
LAUNCH_TIMEOUT = 900
RUN_TIMEOUT = 600

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


async def wait_for_bridge(session, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            async with session.get(f"{BASE}/system_stats", timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(3)
    return False


async def main() -> int:
    import aiohttp

    log = EXT_ROOT / "outputs" / "bridge_smoke.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"launching a session (log: {log})", flush=True)
    # Its own process group, so the whole launcher -- srun, bridge and all --
    # can be torn down together at the end.
    launcher = subprocess.Popen(
        ["bash", str(EXT_ROOT / "run_comfyui_slurm.sh"), str(PORT)],
        stdout=log.open("w"), stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        async with aiohttp.ClientSession() as session:
            up = await wait_for_bridge(session, time.time() + LAUNCH_TIMEOUT)
            check("bridge", up, f"{BASE} answers" if up else "never answered; see the log")
            if not up:
                return 1

            client_id = str(uuid.uuid4())
            payload = json.loads(
                (EXT_ROOT / "example_workflows" / "api" / "01-generate-and-decode.json").read_text())
            events: list[dict] = []
            finished = False
            async with session.ws_connect(f"{BASE}/ws?clientId={client_id}",
                                          heartbeat=20) as ws:
                await session.post(f"{BASE}/prompt",
                                   json={"prompt": payload, "client_id": client_id})
                deadline = time.time() + RUN_TIMEOUT
                while time.time() < deadline:
                    try:
                        message = await asyncio.wait_for(ws.receive(), timeout=30)
                    except TimeoutError:
                        continue
                    if message.type is not aiohttp.WSMsgType.TEXT:
                        # Binary frames are preview images; note the type only.
                        events.append({"type": f"<{message.type.name}>"})
                        if message.type in (aiohttp.WSMsgType.CLOSED,
                                            aiohttp.WSMsgType.ERROR):
                            break
                        continue
                    event = json.loads(message.data)
                    events.append(event)
                    if event.get("type") == "executing" and \
                            event.get("data", {}).get("node") is None:
                        finished = True
                        break

            kinds = [e.get("type") for e in events]
            check("websocket", finished,
                  f"{len(events)} events: {', '.join(sorted(set(kinds)))}"
                  if finished else f"never saw the prompt finish; got {kinds}")

            progress = [e for e in events if e.get("type") == "progress"]
            check("progress", bool(progress),
                  f"{len(progress)} progress events, max "
                  f"{max((e['data'].get('value', 0) for e in progress), default=0)}"
                  f"/{max((e['data'].get('max', 0) for e in progress), default=0)}"
                  if progress else "none arrived -- a first run shows no feedback at all")

            with_images = [
                e for e in events
                if e.get("type") == "executed" and e.get("data", {}).get("output", {}).get("images")
            ]
            total = sum(len(e["data"]["output"]["images"]) for e in with_images)
            check("images", bool(with_images),
                  f"{len(with_images)} nodes reported {total} image(s) to the UI"
                  if with_images else "no 'executed' message carried images -- the "
                                      "preview nodes would stay empty")
    finally:
        print("\ntearing the session down", flush=True)
        try:
            os.killpg(os.getpgid(launcher.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            launcher.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(launcher.pid), signal.SIGKILL)

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 60, flush=True)
    print(f"{len(_results) - len(failed)}/{len(_results)} criteria passed", flush=True)
    if failed:
        print("FAILED: " + ", ".join(failed), flush=True)
        print(f"\n--- last 2000 chars of {log} ---", flush=True)
        print(log.read_text()[-2000:], flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
