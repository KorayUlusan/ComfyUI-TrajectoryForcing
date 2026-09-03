# Contributing

For changing this extension. If you only want to *use* it, the
[README](README.md) is the right document.

Design history and the reasoning behind each pivot lives in [PLAN.md](PLAN.md);
this file is the working knowledge you need before touching the code.

---

## Layout

```
__init__.py             comfy_entrypoint: JAX env, models/ folder, node list
tf_nodes/
  locate.py             where TrajectoryForcing and the weights are
  tf_import.py          the namespace swap (read this before anything else)
  pipeline.py           the only module that touches JAX
  data.py               the three socket payloads
  tokens.py             region clustering + the edit primitive, pure numpy
  render.py             latents/regions/selections -> pictures
  sockets.py            TF_PIPELINE / TF_LEVELS / TF_REGIONS / TF_TOKENS
  nodes_*.py            the nodes themselves
scripts/                workflow generator, smoke tests, measurement
slurm/                  the GPU jobs those run under
tests/                  pytest; no GPU needed
```

The split that matters: `data.py`, `tokens.py` and `render.py` import neither
ComfyUI nor JAX, so the edit semantics are testable in milliseconds. Keep new
logic there and let `nodes_*.py` stay a thin schema-and-wiring layer.

## Tests

```bash
pytest tests                      # 158 tests, no GPU, ~5 s
sbatch slurm/gpu_smoke.sbatch     # the nodes against the real model
sbatch slurm/server_smoke.sbatch  # the workflows through a real ComfyUI server
sbatch slurm/measure_resources.sbatch   # the README's VRAM table
```

`tests/pytest.ini` makes `tests/` the rootdir on purpose: the repository root is
itself an importable package (ComfyUI loads `__init__.py` as the custom node's
module), so a run rooted there builds a Package node for it and tries to import
`__init__` as a top-level module, which fails on its relative imports before a
single test runs.

The three GPU jobs cover different things and all three have caught bugs the
others could not:

- **`gpu_smoke`** runs the nodes directly against the real TF-L model. Its exit
  criteria are in the script's docstring and include a control — the same class
  and seed must reproduce a trajectory bit-for-bit, or the run is void.
- **`server_smoke`** starts a real ComfyUI server and executes the workflows
  through it. This is what covers ComfyUI's validation of the custom socket
  types, and whether the generated workflows are executable rather than merely
  well-formed. It opens a websocket, because a blocked node reports itself there
  and nowhere else.
- **`measure_resources`** produces the README's VRAM table. Re-run it if you
  change what gets loaded.

Write exit criteria into a script's docstring before the run, not after.

---

## Three sharp edges

All three cost a debugging session. Each is commented where it bites; this is
the index.

### 1. TrajectoryForcing and ComfyUI both own a top-level `utils`

TF is a flat repo whose code imports `utils`, `models`, `configs` and
`third_party` by those bare names. ComfyUI ships its own `utils` package,
imported during startup — and once a package is in `sys.modules`, `sys.path` is
never consulted again for it, so TF's `import utils.rae_decoder` resolves
against ComfyUI's `__path__` and fails. Leaving TF's `utils` cached breaks
ComfyUI's `utils.json_util` just as badly in the other direction.

`tf_import.py` swaps the two namespaces rather than merging them. **Every call
into TrajectoryForcing must hold `tf_scope()`** — not just the import — because
the RAE decoder imports `utils.logging_util` and `third_party.rae_decoder`
lazily, from inside functions that first run at decode time.

### 2. Namespace packages get `__file__ = None`, which breaks `sys.modules` walkers

None of TF's directories has an `__init__.py`, so they are all PEP 420 namespace
packages, and CPython gives those `__file__ = None` rather than no `__file__`.
`inspect.getmodule` walks `sys.modules` guarded by `hasattr(m, "__file__")`,
passes that guard, and dies in `inspect.getfile` — but only once the name has
previously been recorded with a real path, which is precisely what the namespace
swap arranges. pydantic calls `getmodule` while building a model class, and
wandb builds pydantic models while TF imports it, so this crashed TF Load
Pipeline inside a running server while the identical import succeeded in a bare
script.

`_bind_namespace_packages` binds every namespace directory in TF's tree up front
with the null `__file__` removed.
`tests/test_tf_import.py::test_a_swapped_module_does_not_break_a_sys_modules_walk`
reproduces the crash.

### 3. Stopping a graph quietly is not what `block_execution` does

Three separate traps, all in `TFTokensFromMask`:

- `block_execution` is only honoured on the branch guarded by
  `result is not None`. Blocking without passing positional args leaves the
  node's outputs empty and ComfyUI's own cache bookkeeping raises `IndexError`.
- A blocker carrying a **message** is reported to the browser as "Node threw an
  error during execution" — and only on the first run, because the second finds
  the node cached, never calls `execute`, and blocks in silence.
- A bare `return ExecutionBlocker(None)` does nothing: `EXECUTE_NORMALIZED`
  turns it into `NodeOutput(block_execution=None)`.

What works: **one `ExecutionBlocker(None)` per declared output**, the reason
carried as the node's own `ui` text, and `has_intermediate_output=True` so the
text survives the cached re-run.

---

## Two invariants the payloads carry

`LevelStack.pipeline` — a trajectory remembers the pipeline that produced it, so
consumers need no wire. `TF Load Levels` restores one without a pipeline, which
is why every consumer keeps an optional `pipeline` socket and calls
`resolve_pipeline`. If you add a node that needs the model, use that helper
rather than a required socket.

`TokenSelection.level` — a selection remembers which level's regions it was
snapped to, or `None` if it was never snapped. Every level shares a token grid,
so a mismatched selection fits perfectly and means something else entirely;
`check_level` is what catches it, and both edit nodes call it. If you add a node
that produces selections from regions, set the level.

## Adding a node

1. Put the logic in `tokens.py` or `render.py` if it is pure; only wiring goes
   in `nodes_*.py`.
2. Add the class to the list in `tf_nodes/__init__.py`.
3. Namespace the `node_id` with a `TF` prefix — ComfyUI has one flat node-id
   space shared with every other extension, and the tests enforce this.
4. Run `pytest tests`. `test_execute_accepts_exactly_the_schema_inputs` checks
   that every schema input matches the `execute` parameter it feeds, which is
   otherwise invisible until someone queues the node in a browser.

Degenerate inputs are results, not crashes — an empty selection, a collapsed
region, a one-region image. Guard the quantity that actually degrades. But do
not paper over a degenerate result either: `TFFeatureEdit` raises on an empty
target rather than passing it through, because an edit that completes having
changed nothing is indistinguishable from an edit that had no effect, and that
is a research error rather than a UI one.

## Regenerating the workflows and figures

```bash
python scripts/make_workflows.py    # after any schema change
python scripts/make_doc_images.py   # after a gpu_smoke run
```

Workflows are generated from the live node schemas rather than hand-written.
LiteGraph stores widget values **positionally**, so a hand-edited file drifts
silently when a widget is added or reordered — it loads with plausible wrong
numbers rather than an error. The generator also validates every combo value
against its live option list, which catches the class of bug where a workflow
names something the server will reject.

Two things it handles that are easy to get wrong by hand: an `Int` input with
`control_after_generate` occupies **two** slots in `widgets_values`, and a
widget wired to a link becomes a converted widget — it keeps its `widgets_values`
slot *and* gains an `inputs` entry naming the widget.

`MarkdownNote` and groups are frontend-only. Notes are excluded from the API
payload (the server has no class by that name), and group bounding boxes are
computed from where their members actually ended up, because a hand-typed box
drifts off its nodes and a group that no longer contains its nodes leaves them
behind when dragged.

## Docs

| file | audience |
|---|---|
| `README.md` | someone who knows ComfyUI and wants this extension |
| `docs/GETTING-STARTED.md` | someone who has never opened ComfyUI |
| `workflows/README.md` | tutorial for the four examples |
| `CONTRIBUTING.md` | this |
| `PLAN.md` | design history, pivots, what the build found |

Keep numbers in one place. The VRAM table comes from
`scripts/measure_resources.py`; job ids and results live in `PLAN.md`. When a
result retires an earlier claim, mark the old one superseded rather than
deleting it — the retraction is the evidence that the criteria were fixed in
advance.

## Commits

Author as `Koray Ulusan <korayulusan2@gmail.com>`. Do not add AI co-authors.
This repository is a thesis artefact and records AI-assisted changes separately,
through the parent repository's `ai-logs/`.
