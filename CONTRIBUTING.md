# Contributing

For changing this extension. To use it, read [README.md](README.md).

---

## A checkout ComfyUI loads

The README installs the published package. To change it you want the git tree
itself on the node path, so edits are live and `git status` is the truth:

```bash
export TF_HOME="$PWD/comfy-tf"

python3.11 -m venv "$TF_HOME/cli"
"$TF_HOME/cli/bin/pip" install comfy-cli
"$TF_HOME/cli/bin/comfy" --workspace "$TF_HOME/ComfyUI" install --nvidia

git clone https://github.com/KorayUlusan/ComfyUI-TrajectoryForcing "$TF_HOME/src"
ln -s "$TF_HOME/src" "$TF_HOME/ComfyUI/custom_nodes/ComfyUI-TrajectoryForcing"

cd "$TF_HOME/src"
cp .env.example .env     # set COMFY_DIR and COMFY_VENV to the two paths above
bash env/setup.sh
./run_comfyui.sh
```

A symlink, not a copy: a copy means editing one tree and running the other, and
the two disagree in exactly the way that wastes an afternoon.

Do not install the registry package into the same workspace. Two copies of these
nodes register the same names and which one wins is not defined.

---

## Layout

```
__init__.py             comfy_entrypoint: JAX env, models/ folder, node list
tf_nodes/
  locate.py             where TrajectoryForcing and the weights are
  tf_import.py          the namespace swap (read this before anything else)
  pipeline.py           the only module that touches JAX
  health.py             startup problems, recorded rather than raised
  doctor.py             `python -m tf_nodes.doctor`, the whole install in one report
  data.py               the three socket payloads
  tokens.py             region clustering + the edit primitive, pure numpy
  render.py             latents/regions/selections -> pictures
  sockets.py            TF_PIPELINE / TF_LEVELS / TF_REGIONS / TF_TOKENS
  nodes_*.py            the nodes themselves
scripts/                workflow generator, smoke tests, measurement
slurm/                  the GPU jobs those run under
tests/                  pytest; no GPU needed
web/                    the one piece of frontend code
.env.example            every environment variable, with its default
.comfyignore            what the registry archive leaves out
install.py              the Manager hook; the only code that touches a user's venv
.github/workflows/      tests on push; publish on a version bump
```

`data.py`, `tokens.py` and `render.py` import neither ComfyUI nor JAX, so the edit
semantics are testable in milliseconds. Put new logic there and keep `nodes_*.py`
a thin schema-and-wiring layer.

---

## Tests

```bash
pytest tests                                        # no GPU, seconds
pytest tests --no-checkout                          # as a fresh runner sees it
./slurm/submit.sh slurm/gpu_smoke.sbatch            # nodes vs. the real model
./slurm/submit.sh slurm/server_smoke.sbatch         # workflows through a real server
./slurm/submit.sh slurm/measure_resources.sbatch    # the README's VRAM table
```

There are four layers, and each is blind to something the next one catches:

| layer | covers | blind to |
|---|---|---|
| `pytest tests` | schemas, edit math, drawing, launchers against stubbed Slurm | anything needing a GPU or a browser |
| `gpu_smoke` | the nodes against the real model, with controls | ComfyUI's execution engine and wire protocol |
| `server_smoke` | the workflows through a real server, on the compute node's loopback | the ncat bridge, which it never crosses |
| `bridge_smoke` | login node to bridge to compute node, as a browser does it | the browser itself |

`--no-checkout` exists because a green local run is not evidence. A dev machine
has a TrajectoryForcing checkout sitting beside the extension, so `tf_repo()`
never reaches its fetch branch and anything that depends on that branch passes
by accident. The flag points both filesystem candidates at an empty directory
and makes a real `git` call an error, which is what a fresh runner looks like.
Tests that genuinely need upstream carry `needs_tf_checkout` and are skipped.

Two guards in `tests/conftest.py` run always, and each maps to a red CI run this
repo actually shipped:

- **No test may leave the environment changed.** A probe that did
  `os.environ.setdefault("TF_NO_AUTO_FETCH", "1")` disabled fetching for every
  test that ran after it. `monkeypatch.setenv` is fine; a bare assignment is not.
- **No unmarked test may leave a checkout in the extension directory.** One that
  did made the "inside the extension" candidate exist for everything downstream,
  and a test asserting the opt-out path stopped raising.

Both were invisible locally and failed on a runner, twenty tests away from the
cause. They now fail on the test that did it, with the reason.

Nothing here drives a browser. When a report is about something only visible on
screen, say which layer proved what rather than implying the whole path is
covered.

Write exit criteria into a script's docstring before the run, not after.
`gpu_smoke` includes a control: the same class and seed must reproduce a
trajectory bit-for-bit, or the run is void.

<details>
<summary>Why <code>submit.sh</code>, why <code>pytest.ini</code>, why a fourth layer</summary>

`submit.sh` rather than plain `sbatch`, because `#SBATCH` lines are comments to
bash and cannot read a variable. That makes partition and QOS, the two settings
that differ on every cluster, the one thing a `.sbatch` cannot carry. The wrapper
reads them from `.env` and puts them on the command line, where they override the
file. Plain `sbatch` works if your default partition has GPUs.

`tests/pytest.ini` makes `tests/` the rootdir on purpose. The repository root is
itself an importable package, since ComfyUI loads `__init__.py` as the custom
node's module. A run rooted there tries to import `__init__` as a top-level module
and fails on its relative imports before a single test runs.

`bridge_smoke` exists because two reports, "no loading bar" and "the image
generates but is not visible", turned out to be one failure seen twice. Both
travel on the websocket. If it drops, the graph still runs, so the server log and
`/history` look healthy while the browser is told nothing. The script checks that
`progress` events and the `executed` message carrying images both arrive. It
allocates a GPU, so it is not part of `pytest`.
</details>

---

## Five sharp edges

Each of these cost a debugging session, and each is commented where it bites.

**1. TrajectoryForcing and ComfyUI both own a top-level `utils`.** Once a package
is in `sys.modules`, `sys.path` is never consulted for it again, so TF's
`import utils.rae_decoder` resolves against ComfyUI's `__path__` and fails.
Leaving TF's `utils` cached breaks ComfyUI just as badly in the other direction.
`tf_import.py` swaps the namespaces rather than merging them. Every call into TF
must hold `tf_scope()`, not just the import, because the RAE decoder imports
`utils.logging_util` lazily from functions that first run at decode time.

**2. Namespace packages get `__file__ = None`.** None of TF's directories has an
`__init__.py`, and CPython gives PEP 420 packages a null `__file__` rather than
none at all. `inspect.getmodule` guards on `hasattr(m, "__file__")`, passes, then
dies in `getfile`. pydantic calls `getmodule` while building a model, and wandb
builds pydantic models while TF imports it, so this crashed TF Load Pipeline
inside a running server while the identical import succeeded in a bare script.
`_bind_namespace_packages` binds every namespace directory up front with the null
`__file__` removed.

**3. Stopping a graph quietly is not what `block_execution` does.** Use one
`ExecutionBlocker(None)` per declared output, carry the reason as the node's own
`ui` text, and set `has_intermediate_output=True`.

**4. The Painter's backdrop is a preview, not the wire.** It resolves the image
behind the brush from the stored UI preview of whatever feeds its `image` slot, so
a node that returns an IMAGE without publishing a preview leaves it blank. That is
why `TF Level Canvas` returns `ui.PreviewImage`. The Painter's image input must
stay socket 0, and its widget exists only under Vue node rendering.

**5. Killing a listener does not free its port.** `ncat --keep-open` forks per
connection and each fork inherits the listening socket, so killing the leader
leaves orphans holding the port. The bridge runs under `setsid` and is killed by
process group. Anything that forks per connection needs the same care.

<details>
<summary>The three traps inside #3</summary>

`block_execution` is only honoured on the branch guarded by `result is not None`.
Blocking without positional args leaves outputs empty, and ComfyUI's cache
bookkeeping then raises `IndexError`.

A blocker carrying a message renders as "Node threw an error during execution",
and only on the first run, because the second finds the node cached and blocks in
silence.

A bare `return ExecutionBlocker(None)` does nothing at all. `EXECUTE_NORMALIZED`
turns it into `NodeOutput(block_execution=None)`.
</details>

---

## Conventions

### Two invariants the payloads carry

`LevelStack.pipeline`. A trajectory remembers the pipeline that made it, so
consumers need no wire. A trajectory from `TF Load Levels` has none, which is why
every consumer keeps an optional socket and calls `resolve_pipeline`. Use that
helper rather than a required socket.

`TokenSelection.level`. A selection remembers which level's regions it was snapped
to. Every level shares a token grid, so a mismatched selection fits perfectly and
means something else entirely. `check_level` catches it. If you add a node that
produces selections from regions, set the level.

### Never ship two widgets where one disables the other

`which` + `level` and `follow_edit` + `level` were both a mode plus a number the
mode silently ignored, and ComfyUI's V3 schema cannot grey a widget out. Fold the
automatic case into the value instead, so that one control always does something.

Rarely-touched settings take `advanced=True`. That halved the visible widgets,
from 41 to 21, without removing anything. Tests assert the ratio, and that the
knobs people actually turn stay visible.

### Which spelling of "automatic"

Three are in use. The choice follows the shape of the domain rather than taste:

| domain | spelling | used by |
|---|---|---|
| closed, small, model-independent | fold it into a combo | `which`: `all levels`, `level 2` |
| closed, length known only at runtime | a combo option opening with `auto` | the checkpoint dropdown |
| open, or bounded by the model | `-1` | `source_level`, resume `level` and `class_id`, `level_override` |

A level is a number, a mode is a dropdown. Reach for `auto_level_input` rather
than a bare `Int.Input` with a `-1` default.

<details>
<summary>Why levels stay numbers, and what <code>-1</code> owes the reader</summary>

An `INT` input can receive a link and a `COMBO` cannot. `TFRegionMap.level`
survived the cut of nine dead sockets precisely because it drives an edit node's
`level`. Making these combos would permanently sever
`TFRegionMap.level -> TFResumeFromLevel.level`.

The level domain is also not closed. `level_override` addresses a model with more
than the four levels every released checkpoint has, so these run to
`MAX_LEVELS - 1` while the visible ones stop at `SHIPPED_LEVELS - 1`. A combo that
preserved that would need sixteen entries plus `auto`.

A sentinel is only obvious to someone who already knows it, so `-1` is stated
three times. `auto_label()` puts `(-1 = auto)` in the `display_name`, which is
visible without hovering in a way a tooltip is not. The tooltip says what auto
does. The node's output says what auto chose, as in "auto: the level the edit
wrote to" versus "set on the node". Tests enforce all three, plus `min == -1` so
no second sentinel creeps in below it.
</details>

### Nodes show their own results

An `info` output is unreachable unless the node that computed it shows it. Once,
42 text and number outputs across the example workflows went nowhere. Every node
that computes a summary now renders it through
`sockets.node_preview(image=..., text=...)`, and needs `has_intermediate_output=True`
or the preview vanishes on the cached re-run. A test enforces the pairing.

### A measurement is not a socket

ComfyUI cannot suggest a destination for an INT or a FLOAT. The frontend's
`slotDefaults.ts` skips every input whose type is in `ComfyWidgets` (INT, FLOAT,
STRING, BOOLEAN, COMBO) unless it declares `forceInput`, and across all of
`comfy_extras` the only such scalar is `floats_strength`, of type `FLOATS`. So
`slot_types_default_out` has no `"INT"` key at all, and a drag from a number
output dead-ends in an empty menu. `PreviewAny` does not help, since it is indexed
under `"*"` and the lookup is by exact type.

An output therefore only exists if a widget somewhere receives it. Nine
measurements were removed for failing that test, and they live in the node's body
and the `report` string instead. `TestEveryScalarOutputDrivesAWidget` requires
every INT/FLOAT output to name the input it feeds.

---

## Dependencies

Never put an installable line in the top-level `requirements.txt`. ComfyUI Manager
pip-installs it into whatever venv ComfyUI is running in, so a line there rewrites
torch and the CUDA libraries under every other custom node in someone's install.
The file is comments-only, and a test enforces that.

Dependencies go in `env/requirements.txt`. `env/setup.sh` installs the same pins
in stages: TF's JAX stack first so ComfyUI's unpinned torch line is already
satisfied, then torch from download.pytorch.org, then the rest. A test keeps the
two lists in step. They cannot be merged, because one requirements file cannot
express three installs from two indexes in a fixed order.

The consequence, which the README states for users, is that a Manager install
registers the nodes and stops at *TF Load Pipeline*, where `check_runtime_deps`
names `bash env/setup.sh`. That is deliberate. A half-finished install is better
than one that breaks the rest of someone's ComfyUI.

---

## Adding a node

1. Pure logic in `tokens.py` or `render.py`, only wiring in `nodes_*.py`.
2. Add the class to the list in `tf_nodes/__init__.py`.
3. Namespace the `node_id` with `TF`, since ComfyUI has one flat node-id space.
4. Run `pytest tests`. `test_execute_accepts_exactly_the_schema_inputs` catches a
   schema input that does not match the `execute` parameter it feeds, which is
   otherwise invisible until someone queues the node in a browser.

Degenerate inputs are results, not crashes: an empty selection, a collapsed
region, a one-region image. Guard the quantity that actually degrades. Do not
paper one over either. `TFFeatureEdit` raises on an empty target rather than
passing it through, because an edit that completes having changed nothing looks
exactly like an edit that had no effect, and that is a research error rather than
a UI one.

---

## Regenerating the workflows and figures

```bash
python scripts/make_workflows.py    # after any schema change
python scripts/make_doc_images.py   # after a gpu_smoke run
```

Workflows are generated rather than hand-written. LiteGraph stores widget values
positionally, so a hand-edited file loads with plausible wrong numbers rather than
an error the moment a widget is added.

Name the output a link comes from, and never count to it:

```python
level=(regions, "level")     # yes
level=(regions, 2)           # refused, with the available names
```

Removing an output shifts every later one, and
`test_every_link_references_real_nodes_and_slots` compares the input's type
against the origin's. That separates `TF_LEVELS` from `IMAGE` but cannot separate
two `INT`s. TF Region Map carried two adjacent INT outputs until `num_regions` was
cut, and wiring the region count into an edit's `level` would have passed the
whole suite while editing the wrong level.

<details>
<summary>What else the generator handles</summary>

An `Int` input with `control_after_generate` occupies two slots in
`widgets_values`. A widget wired to a link becomes a converted widget: it keeps
its `widgets_values` slot and gains an `inputs` entry naming the widget. Combo
values are validated against the live option list, which catches a workflow that
names something the server will reject.

Layout is computed rather than typed. Nodes that display something carry a
realistic `BODY_HEIGHT`, since a `PreviewImage` is about 66px empty and 430px with
an image, and sizing for the empty state is what put content outside its group the
first time anyone ran a graph. Row numbers are an ordering hint. Columns are
stacked so nodes cannot overlap, and group boxes are resolved iteratively, because
a group spanning two columns extends as far as its lowest member in either.

`MarkdownNote` and groups are frontend-only. Notes are excluded from the API
payload, and group boxes are computed from where members actually ended up.
</details>

---

## The one piece of frontend code

`web/tf_token_grid.js` puts a clickable 16x16 grid on *TF Tokens From Coords*. It
exists under one condition that any change has to keep: it may never become the
only way to do something.

The grid writes into the node's own `coords` string rather than replacing it, so
if the file fails to load, or ComfyUI changes an API it leans on, the text field is
still there and typing still works. The whole `onNodeCreated` body is wrapped in a
`try` for the same reason. That condition is the entire reason frontend code is
acceptable here, since failure costs a convenience rather than a feature. Compare
*Painter*, a Vue component that is simply unusable under the classic renderer, and
needed a server-side detector and three paragraphs of docs.

No test covers it and none can, because there is no browser here. Two things
narrow the gap. `tokens.format_coords` is the tested reference implementation of
the notation the JS must reproduce byte-for-byte, with a round-trip property,
without which the grid would silently rewrite what someone typed. And
`server_smoke` checks that ComfyUI found `WEB_DIRECTORY` and serves the file.
Everything past that needs someone to click it, so say that in a PR rather than
implying it was verified.

---

## Releasing

`.github/workflows/publish.yml` publishes to the Comfy Registry. It fires on a
push to main that touches `pyproject.toml`, and nothing else, so the `version`
bump there is the release switch. The registry refuses to re-publish an existing
version, which makes a stray push harmless. There is also a `workflow_dispatch`
trigger for publishing by hand.

Nothing publishes unless the tests pass. `publish.yml` calls `tests.yml` as a
reusable workflow and the publish job `needs:` it. The relative path matters,
because it runs the tests from the commit being published, which a `workflow_run`
trigger could not guarantee.

`install.py` is the riskiest file here, because ComfyUI Manager runs it with the
user's ComfyUI python and it is the only thing in the repo that installs
packages into an environment we do not own. It may only ever add what is
missing. Never upgrade, never downgrade, never touch torch: someone whose
ComfyUI works before installing this must still have one afterwards. Adding JAX
on top of torch was checked on an H100 against a JAX-first venv, both passing
the same five criteria with no `nvidia-*` wheel moving; before widening
`MIN_TORCH` or `CUDA_MAJOR`, run that comparison again rather than reasoning
about it. The pins are asserted against `env/requirements.txt` by
`tests/test_install.py`, which is what stops the two drifting.

That green tick covers lint and the CPU suite. It does not mean the GPU smoke
tests passed. Run `gpu_smoke` and `server_smoke` before bumping the version, and
keep the job ids with the results.

To bump the TrajectoryForcing pin: `TF_REPO_COMMIT` in `tf_nodes/locate.py` is the
commit a fresh install fetches, and a test asserts it is a full sha, because a
branch name there makes every new install track a moving target. Bump it, run
`gpu_smoke`, record the job id. An existing checkout, whether `$TF_REPO` or a
sibling, always wins over the fetch, so your working copy is unaffected.

`.comfyignore` decides what the archive contains. The rule is whether a user of
the nodes would ever open the file, which keeps every `.md` and `docs/img/` and
drops `tests/`, `slurm/`, `scripts/` and `.github/`.

<details>
<summary>Regenerating the logo</summary>

`docs/img/logo/favicon.svg` is the source of truth. `icon.png` is what
`pyproject.toml` points the registry at, by absolute raw-GitHub URL, because the
listing renders outside the repo.

```bash
convert -background none -density 2400 docs/img/logo/favicon.svg -resize 400x400 docs/img/logo/icon.png
python -c "from PIL import Image; p='docs/img/logo/icon.png'; Image.open(p).save(p, optimize=True)"
```

400 is not a preference, it is the registry's documented maximum icon resolution.
Rendering larger does not look sharper in the Manager, it fails the spec. The
density is deliberately far above the output size so the rasteriser has room
before the downsample.

The optimize pass matters. ImageMagick writes several hundred KB for that
gradient and PIL re-encodes the same pixels to about 32 KB. Do not quantise it:
256 colours bands the sheen visibly.

Do not restore `dominant-baseline="central"` to the `<text>`. librsvg ignores it
outright, and rendering with and without it is byte-identical, so it read the `y`
as a baseline and pushed the glyphs 7 units above centre. That looked right in a
browser tab and wrong in every exported PNG. It is positioned by explicit baseline
instead, `y = 26 + capHeight/2`, which is geometry and holds across the font
fallback stack. After any change, measure rather than eyeball:

```python
import numpy as np; from PIL import Image
a = np.asarray(Image.open("docs/img/logo/icon.png").convert("RGBA")).astype(int)
m = (a[...,0]>245)&(a[...,1]>245)&(a[...,2]>245)&(a[...,3]>200)
ys, xs = np.nonzero(m); h, w = a.shape[:2]
print(xs.min(), w-1-xs.max(), ys.min(), h-1-ys.max())   # left right top bottom
```
</details>

---

## Sending a change

Fork, branch, open a pull request. `pytest tests` and `ruff check .` both have to
pass; CI runs them on every push and will not publish a release without them.

If your change touches the model path -- anything under `pipeline.py`,
`tokens.py` or the edit nodes -- say in the PR whether you ran `gpu_smoke`, and
paste the result if you did. Nobody will hold it against you if you have no GPU;
say so and it will be run for you. What causes trouble is a change described as
tested when only the CPU suite was.

Bug reports are far more useful with the output of:

```bash
python -m tf_nodes.doctor
```

which prints the versions, paths and weights that most reports end up asking for
anyway.

## Where the documentation lives

| file | who it is for |
|---|---|
| `README.md` | someone who already uses ComfyUI |
| `docs/GETTING-STARTED.md` | someone who has never opened it |
| `workflows/README.md` | the five example graphs |
| `CONTRIBUTING.md` | you, right now |

Keep numbers in one place. The VRAM table comes from `measure_resources.py`, and
every figure quoted anywhere should trace back to the run that produced it. When
a result retires an earlier claim, mark the old one superseded rather than
deleting it -- the retraction is the evidence that the criteria were fixed in
advance, and it is worth more than the tidier version would be.
