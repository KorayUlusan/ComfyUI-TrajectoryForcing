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
web/                    the one piece of frontend code; see below
```

The split that matters: `data.py`, `tokens.py` and `render.py` import neither
ComfyUI nor JAX, so the edit semantics are testable in milliseconds. Keep new
logic there and let `nodes_*.py` stay a thin schema-and-wiring layer.

## Tests

```bash
pytest tests                      # 352 tests, no GPU, ~5 s
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

## Four sharp edges

All four cost a debugging session. Each is commented where it bites; this is
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

## Two widgets where one disables the other is the shape to avoid

`which` + `level` and `follow_edit` + `level` were both a mode plus a number the
mode silently ignored, and ComfyUI's V3 schema has no conditional widget
visibility to grey the number out. Folding the automatic case into the value
itself leaves one control that always does something:

- `TF Decode Levels` / `TF Latent Preview`: the dropdown names the level
  (`level 2`) instead of a `single level` mode plus a separate number.
- `TF Resume From Level`: `level = -1` means "follow the upstream edit";
  `follow_edit` is gone.
- `TF Feature Edit`: `source_level = -1` means "the level being edited".

`auto_level_input` in `sockets.py` builds that widget. Where a capability only a
deeper model would need had to survive the merge — selecting a level past the
four every released config has — it lives in `level_override`, which **wins
whenever it is set** rather than being conditionally ignored. A widget that can
be ignored is the thing being removed; renaming one would not count.

Rarely-touched settings carry `advanced=True`, which ComfyUI hides behind the
node's advanced toggle (`Comfy.Node.AlwaysShowAdvancedWidgets`). That halved the
visible widgets, 41 to 21, without removing anything. Tests assert both the
ratio and that the knobs people actually turn stay visible.

### `-1` is the one sentinel

Widgets that can work a value out for themselves take `-1` to mean so:
`TF Resume From Level`'s `level` and `class_id`, `TF Feature Edit`'s
`source_level`, the `level_override` on the two preview nodes. One convention,
not three.

A sentinel is only obvious to someone who already knows it, so it is stated in
three places: `auto_label()` puts `(-1 = auto)` in the widget's `display_name`
(visible without hovering, which a tooltip is not), the tooltip says what auto
*does*, and the node's own output says what auto *chose* this run — "auto: the
level the edit wrote to" versus "set on the node". Tests enforce all three, plus
that `min == -1` so no second sentinel can creep in below it.

Reach for `auto_level_input` rather than a bare `Int.Input` with a `-1` default.

#### Which spelling of "automatic" to reach for

`-1` is not the only one here, and the three in use are not arbitrary — but the
rule behind them had never been written down, which is how a fourth gets
invented. It is the shape of the **domain**, not taste:

| domain | spelling | in use by |
|---|---|---|
| closed, small, model-independent | fold the automatic case into a **combo**, so every option is legible without a convention | `TF Decode` / `TF Latent Preview`'s `which` — `all levels`, `level 2` |
| closed, but its length is only known at runtime | a **combo option that opens with `auto`**, prepended to the list | `TF Load Pipeline`'s checkpoint — `AUTO_CHECKPOINT` is `"auto (download TF_L_edit)"` |
| open, or bounded by the *model* rather than the method | **`-1`** plus the three disclosures above | `source_level`, resume `level` and `class_id`, `level_override` |

Two reasons the level widgets are in the last row and should stay there:

- **An `INT` input can receive a link and a `COMBO` cannot.** `TFRegionMap.level`
  is an `INT` output that survived the cut of nine dead sockets precisely because
  it drives an edit node's `level` (see `TestEveryScalarOutputDrivesAWidget`).
  Making the level widgets combos would permanently sever `TFRegionMap.level →
  TFResumeFromLevel.level`, a wiring that is not in a workflow today but is the
  obvious one to want.
- **The level domain is not closed.** `level_override` exists to address a model
  with more than the four levels every released checkpoint has, which is why
  these run to `MAX_LEVELS - 1` while the visible ones stop at
  `SHIPPED_LEVELS - 1`. A combo preserving that needs sixteen entries plus
  `auto`, which is worse than a number.

So: a level is a number. A mode is a dropdown. If a new widget's automatic case
is one of a handful of fixed choices that do not depend on which checkpoint is
loaded, make it a combo and skip the sentinel entirely; otherwise use `-1` and
pay the three disclosures.

## Nodes show their own results

An `info` output is unreachable unless the node that computed it shows it:
forty-two text and number outputs across the example workflows went nowhere,
including `TF Levels Info`, whose entire job is to report.

> This section used to claim stock ComfyUI ships **no** node that displays a
> STRING, "checked against every registered core class". That is wrong —
> `PreviewAny` ("Preview as Text", `comfy_extras/nodes_preview_any.py`) takes
> `IO.ANY` and prints it. The conclusion survives and is the stronger argument
> anyway: a result you have to bolt a second node onto is one nobody reads.

Every node that computes a summary therefore renders it itself, through
`sockets.node_preview(image=..., text=...)`. It returns a merged dict rather
than a `_UIOutput` because `PreviewImage` and `PreviewText` each own only their
own key, and a node with both a picture and a number should show both.

Any node using it needs `has_intermediate_output=True`, or the preview appears
once and vanishes on the next run when the node is served from cache. A test
enforces the pairing.

### A number output must name the widget it drives

**ComfyUI cannot suggest a destination for an INT or a FLOAT.** The suggestion
index is built in the frontend's `extensions/core/slotDefaults.ts`:

```js
if (type in ComfyWidgets) {
  var customProperties = input[1]
  if (!customProperties?.forceInput) continue   // ignore widgets that don't force input
}
```

`ComfyWidgets` holds INT, FLOAT, STRING, BOOLEAN and COMBO, so every scalar
input that is an ordinary widget is skipped. Across all of `comfy_extras` the
one `forceInput` scalar is `floats_strength`, whose type is `FLOATS`. The upshot
is that `LiteGraph.slot_types_default_out` has **no `"INT"` key at all**: drag
from one and the shift-release menu offers nothing. `PreviewAny` does not help —
it is indexed under `"*"`, and the lookup is by exact type string.

STRING escapes this only because `TFSaveReport.text` is `force_input=True`,
which puts one sink in the index.

So: **a measurement is not a socket.** Nine were — `changed_tokens`,
`max_distance`, `arms`, `spread`, `num_regions`, three `count`s and
`num_levels` — and every one of them dead-ended, because you never drive a knob
with a measurement. They were also already in the node's own body and in the
`report` string that `TF Save Report` archives, which is where a number you
*read* belongs. What survives is what drives a widget: `level` into an edit
node's `level`, `class_id` and `seed` into `TF Generate`.

`TestEveryScalarOutputDrivesAWidget` holds the line. Every INT/FLOAT output has
to appear in its `CONSUMERS` map naming the input it feeds, and that input is
checked to exist with a matching type. Adding a convenient `io.Int.Output`
fails the suite until you can say where it goes.

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

### 4. The Painter's backdrop is a *preview*, not the wire

`usePainter.ts` resolves the image behind the brush with
`nodeOutputStore.getNodeImageUrls(node.getInputNode(0))` — the stored UI preview
of whatever feeds its `image` slot. A node that returns an IMAGE but publishes
no preview leaves the Painter blank, which is why `TF Level Canvas` returns
`ui.PreviewImage` with `has_intermediate_output=True`. Two consequences: the
Painter's image input must be socket 0, and the widget itself only exists under
ComfyUI's Vue rendering (`Comfy.VueNodes.Enabled`, readable from
`user/*/comfy.settings.json`, which is how the canvas node knows to warn).

### 5. Killing a listener does not free its port

`ncat --keep-open` forks a child per connection and each fork inherits the
listening socket, so killing the leader leaves the port bound by orphans that
outlive the session. `run_comfyui_slurm.sh` runs the bridge under `setsid` and kills the
**process group**. Anything else that forks per connection needs the same care.

The same background process was also writing to the terminal while `srun --pty`
held it in raw mode, which produced stair-stepped, half-overwritten output — it
logs to a file now.

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

**Name the output a link comes from; never count to it.**

```python
level=(regions, "level")     # yes
level=(regions, 2)           # refused, with the available names
```

Positions are unstable on the output side for the same reason they are on the
input side: removing an output shifts every later one. What makes this worse
than it sounds is that the workflow tests barely cover it —
`test_every_link_references_real_nodes_and_slots` compares the origin output's
type against the link's, and the link's type comes from the *input*, so it
separates a `TF_LEVELS` from an `IMAGE` and cannot separate two `INT`s. TF
Region Map carried two adjacent INT outputs until `num_regions` was cut, and
wiring the region count into TF Feature Edit's `level` would have passed the
whole suite while editing the wrong level — the silent-wrong-answer shape this
extension already has a level check to prevent.

An index now raises in `Graph._slot` at generation time, and
`TestTheWorkflowGeneratorNamesTheOutputsItWires` reads the builders with `ast`
and checks every name against the live schemas, so a stale name fails in the
five-second suite rather than in a workflow that looks fine.

Layout is computed, not typed. Nodes that display something carry a realistic
`BODY_HEIGHT` — a `PreviewImage` is ~66px empty and ~430px with an image, and
sizing for the empty state is what put content outside its group the first time
anyone ran the graph. Row numbers are an ordering hint: columns are stacked so
nodes cannot overlap, and group boxes are resolved iteratively afterwards,
because a group spanning two columns extends as far as its lowest member in
either and two groups can interleave while no two nodes do.

`MarkdownNote` and groups are frontend-only. Notes are excluded from the API
payload (the server has no class by that name), and group bounding boxes are
computed from where their members actually ended up, because a hand-typed box
drifts off its nodes and a group that no longer contains its nodes leaves them
behind when dragged.

## Three test layers, and what each one cannot see

| | covers | blind to |
|---|---|---|
| `pytest tests` | node schemas, edit math, drawing, the launcher scripts against stubbed Slurm | anything needing a GPU or a browser |
| `slurm/gpu_smoke.sbatch` | the nodes against the real model, with controls | ComfyUI's execution engine and its wire protocol |
| `slurm/server_smoke.sbatch` | the workflows through a real server on the compute node's loopback | **the bridge** -- it never crosses it |
| `scripts/bridge_smoke.py` | login node -> ncat bridge -> compute node, as a browser does it | the browser itself |

That fourth one exists because two reports -- "no loading bar" and "the image
generates but is not visible" -- are the same failure seen twice: both travel on
the websocket, and if it drops the graph still runs, so the server log and
/history look perfectly healthy while the browser is told nothing. Nothing
tested that path. It now checks that `progress` events and the `executed`
message carrying output images both arrive. Run it from the login node; it
allocates a GPU, so it is not part of `pytest`.

Nothing here drives a browser. When a report is about something only visible on
screen, say which layer proved what rather than implying the whole path is
covered.

## The one piece of frontend code

`web/tf_token_grid.js` puts a clickable 16x16 grid on *TF Tokens From Coords*.
It is the only JavaScript here, and it exists under one condition, which any
change to it has to keep:

**It may never become the only way to do something.** The grid writes into the
node's own `coords` string; it does not replace it. So if the file fails to
load, ComfyUI changes an API it leans on, or a renderer does not support it, the
text field is still there and typing still works. The whole `onNodeCreated` body
is wrapped in a `try` for that reason -- a broken convenience must not take the
node down with it.

That condition is what makes the frontend code acceptable at all. `PLAN.md`
records the pivot away from a bespoke region-picker widget, and the argument
against it -- frontend code that has to track ComfyUI's frontend releases for a
thesis artefact that must still run in a year -- has not gone away. It is
survivable here only because failure costs a convenience rather than a feature.
Compare *Painter*, which is a Vue component and simply unusable under the
classic renderer: that needed a server-side detector and three paragraphs of
docs. A DOM widget (`addDOMWidget`) renders under both, which is why this is one.

**It is not covered by any test, and cannot be** -- there is no browser or JS
runtime here. Two things narrow that gap:

- `tokens.format_coords` is the tested reference implementation of the
  coordinate notation, and `writeCoords` in the JS must produce byte-identical
  output. `TestFormattingCoordsBack` pins it, including a round-trip property:
  parse(format(mask)) == mask, and formatting twice is stable. Without that the
  grid would silently rewrite what someone typed the moment it was touched.
- `scripts/server_smoke.py` checks ComfyUI found `WEB_DIRECTORY`, lists the
  script under `/extensions`, and serves the right file. That catches a wrong
  path or a renamed directory; it cannot catch a behavioural bug.

Everything past that needs someone to click it. Say so in a PR rather than
implying it was verified.

## Docs

| file | audience |
|---|---|
| `README.md` | someone who knows ComfyUI and wants this extension |
| `docs/GETTING-STARTED.md` | someone who has never opened ComfyUI |
| `workflows/README.md` | tutorial for the five examples |
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
