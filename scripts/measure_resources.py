#!/usr/bin/env python3
"""Measure what this extension actually costs, so the README can state it.

Written because "needs about 8 GB, probably" is the kind of number that gets
copied into a requirements table and is never checked again. Everything printed
here is read from the device or the process, at the point in the workflow where
a user would hit it.

    python scripts/measure_resources.py          # inside a GPU allocation

Reports, in the order a first run encounters them: loading the flow model and
the RAE decoder, sampling one trajectory, decoding all four levels, and holding
two trajectories at once (which the editing workflows do). VRAM is reported both
as the process's own allocations and as what `nvidia-smi` sees, because the two
differ by the CUDA context and by JAX's allocator behaviour, and it is the
second one that decides whether the card is big enough.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

EXT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT_ROOT))

OUT = EXT_ROOT / "outputs" / "resources.json"


def gpu_mib() -> int:
    """What the driver says this process is using -- the number that matters."""
    pid = os.getpid()
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return -1
    for row in rows.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return 0


def host_mib() -> int:
    """Resident set size, for the RAM column."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1


def torch_mib() -> tuple[int, int]:
    import torch

    if not torch.cuda.is_available():
        return 0, 0
    return (
        int(torch.cuda.memory_allocated() / 2**20),
        int(torch.cuda.max_memory_allocated() / 2**20),
    )


def jax_mib() -> int:
    if "jax" not in sys.modules:
        return 0
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
    except (RuntimeError, AttributeError, IndexError):
        return -1
    return int(stats.get("bytes_in_use", 0) / 2**20)


_rows: list[dict] = []


def record(stage: str) -> None:
    torch_now, torch_peak = torch_mib()
    row = {
        "stage": stage,
        "gpu_process_mib": gpu_mib(),
        "jax_in_use_mib": jax_mib(),
        "torch_now_mib": torch_now,
        "torch_peak_mib": torch_peak,
        "host_rss_mib": host_mib(),
    }
    _rows.append(row)
    print(
        f"{stage:<34} gpu {row['gpu_process_mib']:>6} MiB   "
        f"(jax {row['jax_in_use_mib']:>5}, torch {torch_now:>5}/{torch_peak:>5} peak)   "
        f"ram {row['host_rss_mib']:>6} MiB",
        flush=True,
    )


def main() -> int:
    from tf_nodes.locate import register_model_folder
    from tf_nodes.tf_import import configure_jax_env

    configure_jax_env()
    register_model_folder()
    record("baseline (nothing loaded)")

    from tf_nodes.nodes_edit import TFFeatureEdit, TFResumeFromLevel
    from tf_nodes.nodes_pipeline import TFDecode, TFGenerate, TFLoadPipeline
    from tf_nodes.nodes_regions import TFRegionMap, TFTokensFromCoords

    pipeline, _ = TFLoadPipeline.execute(
        checkpoint="auto (download TF_L_edit)", config="edit_env_config.yml", warmup=False
    )
    record("TF Load Pipeline (no warmup)")

    target, = TFGenerate.execute(pipeline=pipeline, class_id=213, seed=592)
    record("TF Generate (first, compiles XLA)")

    TFDecode.execute(pipeline=pipeline, levels=target, which="final level only",
                     level=3, label_levels=False)
    record("TF Decode, final level only")

    TFDecode.execute(pipeline=pipeline, levels=target, which="all levels",
                     level=0, label_levels=False)
    record("TF Decode, all 4 levels")

    # The editing workflows hold two trajectories and re-sample from a level;
    # this is the high-water mark a real session reaches.
    source, = TFGenerate.execute(pipeline=pipeline, class_id=207, seed=592)
    regions, _, _ = TFRegionMap.execute(levels=target, level=2, cosine_threshold=0.9, size=512)
    tgt, _ = TFTokensFromCoords.execute(coords="7,7", levels=target, regions=regions)
    src, _ = TFTokensFromCoords.execute(coords="6,6:9", levels=source, regions=None)
    edited, _ = TFFeatureEdit.execute(
        levels=target, level=2, target_tokens=tgt, source_tokens=src,
        source_mode="region mean", strength=1.0, source_level=2, source_levels=source,
    )
    resumed, _ = TFResumeFromLevel.execute(
        pipeline=pipeline, levels=edited, level=2, follow_edit=True, class_id=-1, seed=592
    )
    TFDecode.execute(pipeline=pipeline, levels=resumed, which="all levels",
                     level=0, label_levels=False)
    record("full edit workflow (2 trajectories)")

    peak = max(r["gpu_process_mib"] for r in _rows)
    peak_ram = max(r["host_rss_mib"] for r in _rows)
    print("\n" + "=" * 78, flush=True)
    print(f"peak VRAM for this process: {peak} MiB ({peak / 1024:.1f} GiB)", flush=True)
    print(f"peak host RAM:              {peak_ram} MiB ({peak_ram / 1024:.1f} GiB)", flush=True)
    print("\nNote: ComfyUI itself adds its own CUDA context and any other models a", flush=True)
    print("workflow loads; these numbers are TrajectoryForcing's share only.", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": _rows, "peak_vram_mib": peak, "peak_ram_mib": peak_ram}, indent=2))
    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
