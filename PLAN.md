# ComfyUI-TrajectoryForcing — Plan

A ComfyUI custom-node extension exposing Trajectory Forcing's (TF) coarse-to-fine
generation, per-level preview, and interactive editing as a node graph. This file
is the plan and its pivots, same convention as `TrajectoryDreamer/PLAN.md` — read
`README.md` for what exists and how to run it, `TrajectoryForcing/README.md` and
the TF paper (`_koray/Trajectory-Forcing-2606.22527v1.pdf`) for the method.

**Scope: TF only.** TF-2.0 (text conditioning) has zero trained checkpoints as of
this writing (see `_latex/kickoff-slides/TF-diag.tex`, Chapter 2 TL;DR) — a
`TF Text Encode` node would have nothing to load against, so it is deferred until
TF-2.0 produces a checkpoint. Trajectory Dreamer (3D) is Phase 2 and not touched
here at all.

## Status — 2026-09-04

Steps 1–5 of the original build order are done and verified on an H100; the
open questions below are all answered. Five browser sessions have driven it, and
each is written up below — every one produced defects that reading the code had
not. What has *not* happened is an edit made for a **research** reason rather
than to exercise the tool: chosen because the answer matters, run end to end,
swept, and written into a results table. That is the next thing, and everything
in `Next` stays a guess until it happens.

| build step | state |
|---|---|
| 1. Load + Generate + Decode | done — `gpu_smoke` 8/8, job 449837 |
| 2. Region picker, read-only | done, but not as a custom widget — see the pivot below |
| 3. Feature edit (same-canvas and cross-canvas) | done — `TF Feature Edit`, both source modes |
| 4. Resume, closing the loop | done — `TF Resume From Level` |
| 5. Shape edit | done — `TF Shape Edit`, region-mean reassignment |

Verification, all reproducible from this repo:

- `pytest tests` — 352 tests, no GPU, ~5 s. Edit math, payloads, drawing, and
  every node's schema and `execute` against a stub pipeline.
- `slurm/gpu_smoke.sbatch` — the nodes against the real TF-L model. **8/8,
  job 449988**, including two controls: same class + seed reproduces the
  trajectory bit-for-bit, and the sweep's arm 0 reproduces the explicit
  edit-and-resume chain bit-for-bit. Load 14.4 s warm, generate 0.0 s, decode
  0.2 s, resume 2.1 s, a three-seed sweep 0.8 s. (Superseded: 8/8 on job 449980,
  which predates the nine-output cut; 7/7 on job 449604, which predates
  `TF Compare Levels` and `TF Sweep Edit`.)
- `slurm/server_smoke.sbatch` — the workflows through a real ComfyUI server.
  **13/13, job 449990**, 20 nodes registered, all five workflows
  executed. This is the only check that a node running several re-samples in one
  `execute` survives ComfyUI's execution engine rather than a direct call, and
  that the sweep's table reaches the client rather than only its images.
  (Superseded: 13/13 on job 449981, which predates the nine-output cut; 9/9 on
  job 449655, which predates workflow 05. Job 449989, between them, is the 12/13
  that found the script's own hardcoded output list — see the section on the nine
  outputs.) The first attempt (449598) was 2/6 and is
  what found both the namespace bug below and a workflow naming an ImageNet
  class string one word off. The criteria covering the painter
  workflow's quiet stop were added after the first browser session and took five
  runs to get right (449615, 449616, 449617, 449618, 449655) — every failure was
  a real defect, and the last two only became visible once the test ran the
  workflow twice and watched the websocket.

- `scripts/bridge_smoke.py` — the login node's bridge to the compute node,
  driven as a browser drives it. **4/4.** The only layer that crosses the
  `ncat` bridge, and the one that disproved a plausible diagnosis rather than
  confirming one. Allocates a GPU, so it is run by hand, not by `pytest`.

## Pivots

### The region picker is not a custom LiteGraph widget

The original plan had step 2 building a bespoke canvas widget: 16×16 grid
overlay, click → region-id lookup. That is not what got built.

ComfyUI 0.34 ships a **`Painter`** node that takes an optional `IMAGE` input and
returns `IMAGE` + `MASK`. So the interaction decomposes into pieces that already
exist: `TF Level Canvas` renders the level at 512px with the token grid and
region boundaries drawn on, `Painter` supplies the brush, and
`TF Tokens From Mask` reduces the painted mask back to tokens — snapping to
whole cosine regions when a region map is wired in, which is the editing env's
"cluster select" without a line of JavaScript.

This is better than the plan, not just cheaper. A brush answers step 5's open
question (discrete clicks vs. a real stroke) in one move, since shape edits want
strokes along a boundary and feature edits want blobs. It also means there is no
frontend code to keep in step with ComfyUI's frontend releases, which for a
thesis artefact that has to still run in a year is the deciding argument.
`TF Tokens From Coords` covers what a brush cannot: an experiment that has to be
written up needs a selection someone else can reproduce exactly.

The cost is one round-trip — you run the graph once to get a canvas, paint, then
run again. Acceptable; revisit only if that becomes the thing that annoys.

### Both edit nodes share one primitive, as planned

`z̃ᵢ = f_src` for the selected tokens, in `tokens.py`. Feature edit sources
`f_src` from a selection (this trajectory or a second one); shape edit sources it
from the region absorbing the tokens. `TF Resume From Level` is downstream of
either and does not care which produced its input. Two additions the plan did not
have: a `strength` lerp (the paper's edit is 1.0), and a `region mean` vs
`token cycle` choice — the former is the paper's `f_src`, the latter reproduces
the editing env's token-for-token copy.

### Resume goes through the public API, not a private one

The plan assumed reusing `Pipeline.edit`'s machinery. It turned out `edit` *is*
the resume operation with a token exchange bolted on the front, so
`pipeline.resume()` calls it with source == target and one token copied onto
itself. The edit is a no-op and only the resume remains. Nothing private is
touched and none of TF's sampling math is re-derived here.

## Answers to the original open questions

**ComfyUI's venv — separate, and torch had to move off TF's pin.**
`~/.venvs/comfyui-tf`, built by `env/setup.sh`. TF's JAX stack goes in first so
ComfyUI's unpinned `torch` line is already satisfied. The one real conflict:
ComfyUI 0.34's `comfy-kitchen==0.2.31` registers a torch custom op annotated
`list[int]`, which torch 2.6's `infer_schema` rejects — `import comfy.quant_ops`
fails and ComfyUI does not start at all. Resolved at **torch 2.8.0+cu128**
(cu128, not cu130, to keep the same CUDA major as jax 0.4.36). Nothing on TF's
inference path is version-sensitive and this is inference only, but it is a
deviation from `TrajectoryForcing/requirements.txt` and belongs in any writeup of
a number produced through these nodes.

`wandb` is also required, which `editing_env/requirements.txt` does not list:
`utils/logging_util.py` imports it at module scope and `utils/ckpt_util.py` —
which `Pipeline.__init__` imports — pulls that in.

**Checkpoint location — ComfyUI's `models/` convention, for the flow checkpoint
only.** `models/trajectory_forcing/` is registered with `folder_paths` and gives
the loader node a dropdown; `auto` downloads `TF_L_edit` into it. The RAE decoder
stays wherever the TrajectoryForcing checkout has it (`checkpoints/rae/`), since
`scripts/download_models.sh` and the editing env both put it there and a second
2 GB copy buys nothing. `TF_RAE_ROOT` overrides.

**Shape-edit interaction — brush.** See the pivot above.

**`XLA_PYTHON_CLIENT_MEM_FRACTION` — no value, and the question was malformed.**
`MEM_FRACTION` only sizes the *preallocated* block; with `PREALLOCATE=false` it
does nothing. Growth-on-demand and a hard cap are alternatives, not a pair. The
default is growth-on-demand, which is what `comfy.model_management` can cope with
(it cannot see JAX's allocations either way, but at least the remaining VRAM is
real). `TF_XLA_MEM_FRACTION=0.3` flips to preallocation with that cap and passes
ComfyUI a matching `--reserve-vram`. TF-L plus the ViT-XL decoder is a few GB on
an 80 GB card, so this is only worth reaching for alongside something big.

## What the build found that the plan did not anticipate

Both were found by running the thing, not by reading it, and both are written up
in `README.md` under "Two things worth knowing if you touch this".

1. **TF and ComfyUI both own a top-level `utils`** (also `models`, `main`).
   ComfyUI's is a real package imported at startup, so once it is in
   `sys.modules` TF's `import utils.rae_decoder` resolves against ComfyUI's
   `__path__` and fails — `sys.path` is never consulted for an already-imported
   package. `tf_import.py` swaps the two namespaces rather than merging them, and
   every call into TF holds that scope, because the RAE decoder imports
   `utils.logging_util` and `third_party.rae_decoder` lazily at decode time.

2. **Namespace packages get `__file__ = None`, which breaks `sys.modules`
   walkers.** None of TF's directories has an `__init__.py`. `inspect.getmodule`
   walks `sys.modules` guarded by `hasattr(m, "__file__")`, passes on a null
   `__file__`, and raises in `inspect.getfile` — but only once that name has been
   recorded with a real path, which is precisely what the namespace swap
   arranges. pydantic calls `getmodule` while building a model class; wandb
   builds pydantic models while TF imports it. So TF Load Pipeline died inside a
   running ComfyUI server while the identical import succeeded in a bare script.
   This is the bug the server-level smoke test exists to have caught, and it
   would have been near-impossible to find from the UI.

## From the first session in the browser (2026-09-03)

Two things came straight back from actually launching it, which is the argument
for doing that before anything on the Next list.

**`sbatch` was the wrong launcher.** It detaches from the shell, so there is no
log to watch and quitting leaves the H100 allocated until the walltime expires;
and the tunnel instructions it printed were wrong twice over (they SSH'd into
the compute node and forwarded it to itself, and login-to-compute SSH wants a
password here because `~/.ssh/authorized_keys` holds a different key from
`~/.ssh/id_ed25519.pub`). Replaced with `./run_comfyui_slurm.sh`: `srun` in the foreground so
the log streams and Ctrl-C cancels, plus an `ncat` bridge on the login node so
no SSH hop to the compute node is needed at all.

**The painting workflow's first run looked like a crash.** It is two-pass by
construction -- run to get the canvas, paint, run again -- but with nothing
painted the empty selection travelled two nodes downstream and surfaced as a
`ValueError` traceback from TF Feature Edit. `TF Tokens From Mask` now stops the
graph itself with an instruction, and distinguishes the three reasons a
selection comes out empty (nothing painted / too thin for `coverage` / no region
cleared `region_overlap`), since they need different fixes. The alternative --
passing an empty selection through as a no-op -- was rejected: an edit that
completes having changed nothing is indistinguishable from an edit that had no
effect, which is a research error and not just a UI one.

Stopping a ComfyUI graph *quietly* took three attempts, each caught by putting
it back in front of a browser. All three are commented where they bite.

1. `block_execution` is only honoured on the branch guarded by `result is not
   None`, so blocking without passing placeholder positional args leaves the
   node's outputs empty and ComfyUI's own cache bookkeeping dies with an
   `IndexError`.
2. A blocker carrying a *message* is reported to the browser as "Node threw an
   error during execution" — and only on the first run, because the second finds
   the node cached, never calls `execute`, and blocks in silence. Alarming and
   inconsistent, which is worse than either alone. The fix is one
   `ExecutionBlocker(None)` per declared output (a bare `return
   ExecutionBlocker(None)` does not work — `EXECUTE_NORMALIZED` turns it into
   `block_execution=None`, i.e. no block at all), with the reason carried as the
   node's own `ui` text and `has_intermediate_output=True` so it survives the
   cached re-run.
3. A blocked node reports itself over the websocket only — `/history` still says
   the prompt succeeded — so the server-level test had to open a websocket to
   see what the browser sees, and to run the workflow *twice*, since a single
   run cannot distinguish "raises then goes quiet" from "quiet throughout".

## From the second session (2026-09-03, same day)

Launching it again turned up three more things, all of them presentation rather
than mechanism -- which is what a working tool's problems look like.

**`run_comfyui_slurm.sh` said too little at the start and too much at the end.** It now
opens with what is about to happen and, specifically, the exact log line to wait
for (`To see the GUI go to:`), because the port genuinely is not open before it
and the silence in between reads as a hang. On Ctrl-C it now says it is
cancelling, waits for Slurm to actually release the allocation, and only warns
if it did not -- the previous version printed a WARNING immediately, during the
normal few-second teardown, which made a clean exit look like a fault.

**The workflows were a wall of unlabelled boxes.** They now carry ComfyUI groups
(numbered, in execution order) and a `MarkdownNote` on the left of each,
explaining what the workflow does and what is worth changing. The generator
computes group bounding boxes from where the member nodes actually ended up,
because a hand-typed box drifts off its nodes and a group that no longer
contains its nodes leaves them behind when dragged.

**One README could not serve both audiences.** Split into `README.md` (someone
who knows ComfyUI), `docs/GETTING-STARTED.md` (someone who has never opened it),
`workflows/README.md` (a tutorial for the five examples) and `CONTRIBUTING.md`
(the sharp edges). The requirements section now carries measured numbers rather
than a guess: `scripts/measure_resources.py` reports **6.6 GiB peak VRAM** and
5.3 GiB host RAM for the full editing workflow (job 449659), which puts the
practical floor at an 8 GB card.

## From judging it as a user (2026-09-03)

A pass over the node set asking what a user actually has to do, rather than
whether it works. Three things came out, two of them measured rather than felt.

**One silent-wrong-answer bug.** A selection snapped to level 2's regions could
be applied at level 1 with no complaint. Every level shares a token grid, so the
shapes match and the result looks plausible while being the wrong region
entirely. `TF Shape Edit` had always checked this; `TF Feature Edit` had not --
the inconsistency is what made it easy to miss. `TokenSelection` now records the
level whose regions it was snapped to, and both edit nodes check it. Combining
two selections from different levels is refused for the same reason.

**`level` was typed four times in one workflow** -- TF Region Map, TF Feature
Edit's `level` and `source_level`, TF Level Canvas -- with nothing enforcing
agreement. "Try editing at level 1" therefore meant finding and changing all
four, and missing one produced the bug above. TF Region Map now outputs its
`level`, and the workflows wire it, so the widget exists in one place.

**Six `TF_PIPELINE` wires crossed workflow 02.** A trajectory always comes from
a pipeline, so `LevelStack` now carries it and the consumers find it there.
Measured: 6 wires down to 2 in workflow 02, 5 to 2 in 03, 3 to 1 in 01 and 04.
The socket stays optional, because a trajectory restored by TF Load Levels has
no pipeline to carry.

Also from the same pass: coordinates now readable rather than countable (row and
column numbers on every grid preview), `TF Tokens From Coords` and `TF Tokens
From Mask` work without a `levels` wire, and `TF Levels Info` reports the class
*name*.

The last of those small items needed a second attempt. `TF Decode Levels` and
`TF Latent Preview` each carry a `level` widget that does nothing unless `which`
is `single level`, and the first fix was to reword the tooltip -- which changes
nothing, because nobody reads a tooltip before turning a knob. ComfyUI's V3
schema has no conditional widget visibility (only a static `hidden` flag), so
the widget cannot be greyed out. What works is for the node to say it is
ignoring you, in its own body, at the moment it happens -- the same
`ui.PreviewText` route the painting notice uses, with
`has_intermediate_output=True` so it survives the cached re-run. It stays quiet
when the widget is still at its default, since that is not a change anyone made.

**New: `TF Compare Levels`.** Two trajectories in, tokens-changed per level and
a heatmap of where out. The question every edit prompts is "did that do
anything", and the only previous answer was two pictures and your judgement. An
edit at l* should show zero change below l*, the selected tokens at l*, and
diffuse change above -- anything else is the interesting kind of wrong, and now
visible as a number.

Not done, and still the biggest gap: **a sweep node**. Running the same edit
across seeds or across l* is the thing a graph should make cheap and the Gradio
app cannot, and it still means duplicating the graph by hand.

**Superseded the same day** by "The sweep node" below: `TF Sweep Edit` and
workflow 05. The paragraph is kept because the gap it names is what the sweep's
design answers, and the three forks recorded there only make sense against it.

## Third browser session (2026-09-03)

Three reports, all about workflow 03, all traced to the frontend rather than
guessed at by reading the ComfyUI source map.

**The Painter needs "Node 2.0".** Its widget is registered as
`PAINTER: transformWidgetConstructorV2ToV1(usePainterWidget())` and only exists
under Vue node rendering; the classic renderer shows the string `node2only`
("Node 2.0 only") instead of a brush. The setting is `Comfy.VueNodes.Enabled`
and -- usefully -- it is stored server-side in
`user/*/comfy.settings.json`, so `TF Level Canvas` reads it and says exactly
where to click, but only for the users who have it off. Documented in the
workflow note, `workflows/README.md` and the README troubleshooting list.

**The Painter showed no image to paint over,** which is the difference between
painting and painting blind. `usePainter.ts` resolves its backdrop with
`nodeOutputStore.getNodeImageUrls(node.getInputNode(0))` -- the *stored preview*
of whatever feeds its image slot, not the tensor on the wire. `TF Level Canvas`
returned an image but published no preview, so there was nothing to find. It now
returns `ui.PreviewImage` with `has_intermediate_output=True`. A consequence
worth knowing: the Painter's image input must stay socket 0, which
`tests/test_workflows.py` now pins.

**Nodes overflowed their groups once the graph had run.** A `PreviewImage` is
~66px tall while empty and ~430px with an image, and the generator sized every
node for its empty state -- so groups were 22px of slack around content that
grew by 400, and in workflow 02 the previews landed on top of each other. Fixed
at the source rather than by nudging coordinates: nodes that display something
carry a realistic `BODY_HEIGHT`, columns are stacked so nodes cannot overlap
(row numbers became an ordering hint rather than a literal position), a group
transition reserves room for both borders and the title bar, and group boxes are
resolved iteratively afterwards -- a group spanning two columns extends as far
as its lowest member in *either*, so two groups can interleave while no two
nodes do. `tests/test_workflows.py` checks all of it, plus that no link dangles
and no virtual note reaches the API payload.

## The bridge leaked its own port (2026-09-03)

Reported as `Ncat: bind to 127.0.0.1:8188: Address already in use` on launch,
with the log stair-stepped and half-overwritten. Suspected VS Code's port
forwarding; it was neither VS Code nor the port. Thirty-four orphaned `ncat`
processes were still bound to 8188, all forwarding to `mlcbm009`, a node whose
job had ended long before.

`ncat --keep-open` forks a child per connection and **each fork inherits the
listening socket**, so `kill $bridge_pid` killed the leader and left every fork
holding the port. Reproduced deliberately: with five connections open, killing
the leader leaves five processes and the port bound; killing the process group
leaves none and releases it. The bridge now runs under `setsid` -- its own
process group -- and cleanup kills the group.

Three things came with it. `run_comfyui_slurm.sh` sweeps stale bridges of its own at
startup, identified by the compute node in their command line having no running
job of ours, so an existing mess clears itself rather than needing a manual
`pkill`. If the port is genuinely taken by something else it steps to the next
free one and says so, rather than dying inside a log nobody is reading. And the
bridge now writes to a file instead of the terminal: `srun --pty` puts the
terminal in raw mode, and a background writer into it is what produced the
stair-stepped output -- the garbling and the bind error were the same bug seen
from two angles.

## The sweep node (2026-09-03)

The gap named at the end of the second browser session, closed. `TF Sweep Edit`
runs feature edit → resume → measure once per value of one axis (`seed`,
`level (l*)`, `strength`), with everything else pinned, and emits a table, a
contact sheet and one arm on a `TF_LEVELS` output. Workflow 05 is that graph.

Three decisions are worth recording, because each was a fork in the design:

**Every arm gets its own baseline.** The obvious implementation measures each
arm against the input trajectory. That is wrong for the axis the node exists
for: resuming from an edited canvas is still sampling, so a seed sweep measured
that way reports the seed's effect and the edit's added together, and the
biggest number wins for the wrong reason. Each arm is instead measured against
the *same* trajectory resumed at the same level with the same seed and no edit
— one extra resume per arm, which is ≤3 NFE and cheap next to the decode. The
baseline is cached per `(level, seed)`, so a strength sweep pays for one rather
than one per arm. Confirmed on hardware: the four-arm strength case issues five
resumes, not eight.

**The spread across arms is the output that justifies the node.** Per-arm
"tokens changed" is what `TF Compare Levels` already gives, four times over.
The number only a sweep can produce is the mean pairwise distance *between*
arms: whether the axis moves the outcome at all. On job 449818 a three-seed
sweep gave mean edit distance 0.066 against arm-to-arm spread 0.027 — the edit
is ~2.5× the seed noise, so the single-seed result in workflow 02 was worth
trusting. Had the ratio gone the other way it would not have been, and nothing
in the previous toolset could have said so.

**Sweeping l\* breaks the level check, deliberately.** A selection snapped to
level 2's regions is refused at another level everywhere else in the extension,
and that guard is right — the token grid is the same size at every level, so the
edit would otherwise land on the wrong region silently. But "the same edit at
every level" has to mean the same *token set*, so the level axis is the one
place the binding cannot hold. It reports the caveat instead of refusing; the
other two axes keep the check. Refusing would have made the axis unusable and
quietly dropping the check would have made three other axes unsafe.

The sweep carries its own control, in `gpu_smoke.py` criterion 7: the arm whose
seed matches the explicit edit-and-resume chain must reproduce that chain's
trajectory bit-for-bit. A loop that re-reads the source at the wrong level, or
shifts a seed by one, still produces a plausible table — the table is not
self-checking, so something else has to be.

**Superseded (2026-09-03):** the sweep now covers shape edits as well. The
original note said a shape edit had "no meaningful axis but the seed" and then
declined to offer the seed -- a sentence conceding the case it went on to
refuse. Wiring a region map in switches the edit; both reduce to the same
primitive and differ only in where f_src comes from, so the loop is unchanged.
Only l* is genuinely impossible, because a region map describes exactly one
level, and that is now refused with the reason rather than silently unsupported.

## A UX pass over the whole thing (2026-09-03)

Six findings, ranked by what a user actually hits. Five were fixed; the sixth
was a design choice and went back to the user.

**The one that was reported as a bug first.** "I clicked Run on workflows 1 and
5 and nothing happened." That session left **no evidence at all** — `run_comfyui_slurm.sh`
streamed to a terminal and then `rm -f`'d its own bridge log, and
`run_comfyui.sh` `exec`'d ComfyUI with no log file. Nothing on disk, so nothing
to read. Both logs now persist under `outputs/comfyui_logs/`, last 20 of each,
and `run_comfyui_slurm.sh` prints both paths on exit. ComfyUI's own `--verbose INFO FILE`
does the writing rather than a `tee`, because a pipe makes stdout not-a-terminal
and `srun --pty` needs a real one on both ends or Ctrl-C stops reaching ComfyUI.

The likely cause, unprovable after the fact: `TF Load Pipeline` is the slowest
thing here and was the only slow thing with **nothing to watch**. It now drives a
three-stage progress bar (restore → compile → decoder build) and reports how
long it took. A cached load completes the bar rather than leaving it a third of
the way across, which reads as stuck.

**A defect in the sweep's own surface.** `axis` and `values` are two widgets that
must agree, and the default `values` only fits the default `axis` — so the first
failure most people meet is switching to `strength` and getting *"Strength 2.0 is
outside 0..1"*, which is true and says nothing about which widget is wrong. Every
range error now names the coupling and gives an example.

**The numbers could not leave the graph** — the biggest functional gap once the
sweep existed. `TF Save Levels` archived latents and history; the sweep and
compare tables lived only as text in a node body you had to select and copy. So
the tool could measure an edit and not archive the measurement, which by this
repo's own rule about untraceable numbers is the wrong half to be missing. **New:
`TF Save Report`**, writing markdown fenced so the columns survive, stamped with
the class, seed and edit history that produced them, appending so a session's
runs accumulate in one comparable file. Wired into workflow 05.

**The contact sheet was not one.** The `sheet` output was a batch of five 384px
frames, which ComfyUI renders inside a ~320px node and `SaveImage` writes as five
separate files — so the side-by-side comparison the node exists for was available
in neither place. `sheet_layout` (advanced) now defaults to one stitched image, a
row up to six arms and a near-square grid beyond, with the batch still available
because it is the only shape that saves one file per arm. Verified at real size:
2072×544 for four 512px frames. Deliberately *not* applied to `TF Decode Levels`
(a sequence, often wanted individually) or `TF Compare Levels` (per-level tiles);
only a sweep's arms are meaningless apart.

Two smaller ones: `TF Sweep Edit` had five visible widgets where every other node
has at most two, so `strength` moved to advanced; and `TF Load Levels`'s dropdown
now says the list is rebuilt when node definitions are sent, so press R after
saving. (`GET_SCHEMA` does re-run `define_schema` per `/object_info`, checked
rather than assumed — the list was never actually stale, only silent about it.)

**`gpu_smoke.py` had gone stale twice** — once when `which` + `level` merged, once
when the sweep gained `sheet_layout` — and both times it failed minutes into a
queued GPU job. It calls node `execute` methods by hand and nothing checked those
call sites. `TestTheSmokeScriptsCallNodesCorrectly` now reads them out of the
source with `ast` and checks them against the live schemas in the five seconds
the rest of the suite takes. Confirmed to fail on the real bug before being kept.

## The pivot on the region picker, revisited (2026-09-03)

The plan's step 2 was a bespoke clickable grid widget; PLAN records dropping it
for Painter, on the grounds that frontend code has to track ComfyUI's frontend
releases and this artefact has to still run in a year. Asked to revisit it, that
argument turned out to be right about *a widget that is the only way to do
something* and wrong about a widget that decorates one.

`web/tf_token_grid.js` puts a clickable 16x16 grid on TF Tokens From Coords. It
writes into that node's existing `coords` string rather than replacing it, so:
the selection stays a text value someone can paste into a writeup (the entire
reason that node exists over the brush); no backend, socket or workflow changed;
and **if the file fails to load, typing still works.** That last property is what
makes the bet cheap, and it is the condition any change here has to keep.
Contrast Painter, which is a Vue component and simply unusable under the classic
renderer -- that cost a server-side detector and three paragraphs of docs.

**No v1/v2 split, and that was checkable rather than a guess.** Core's own
`AUDIO_UI` widget calls `addDOMWidget` unconditionally via `getCustomWidgets`, in
a ComfyUI that still ships the classic renderer, and nothing near `addDOMWidget`
branches on the renderer -- DOM widgets predate Vue nodes. One implementation
covers both. Core has no click-a-point affordance to reuse, incidentally: its own
spatial node (`comfy_extras/nodes_wanmove.py`) takes four float sliders.

**It is the first thing in this repo no test covers**, and that is stated rather
than papered over. There is no browser here. Two things narrow the gap:
`tokens.format_coords` is a tested reference implementation of the notation that
the JS must reproduce byte-for-byte (round-trip property: parse(format(m)) == m,
and formatting twice is stable -- without it the grid silently rewrites what was
typed the moment it is touched), and `server_smoke` now checks ComfyUI found
`WEB_DIRECTORY`, listed the script under `/extensions`, and served the right
file. Behaviour still needs someone to click it.

## A browser session, and three wrong turns (2026-09-03)

Four reports from one sitting. All four were real, and the record of how they
were diagnosed matters more than the fixes, because three of my first
explanations were wrong and the pattern in them is worth keeping.

**`./serve.sh` exited silently, printing nothing.** Mine, and freshly
introduced: an empty glob makes `ls` exit 2, `2>/dev/null` hides the message but
not the status, and `set -o pipefail` carries it out through `set -e`. The line
was the log-pruning added the same day *to diagnose a silent failure*. It was in
both launcher scripts, so the Slurm route died on the login node and again on
the compute node. `bash -n` cannot see this; only running the script can.
`tests/test_scripts.py` now runs the real launcher against stubbed `srun`,
`squeue`, `ncat`, `ss`, `setsid` and `pgrep`, so every line of setup executes.
Reintroducing the bug fails five of its tests. Renamed to
`run_comfyui_slurm.sh` at the same time, so it pairs with `run_comfyui.sh`
rather than hiding that it asks Slurm for a GPU.

**"Workflow 1 gives me level 0, it should give all levels."** ComfyUI grids a
multi-image output until you *click* one, which pins the node to that frame via
`imageIndex`. The fix was to stitch frames meant to be seen together into one
contact sheet -- which had already been built for the sweep two hours earlier
and **deliberately not applied here**, on the reasoning that decoded levels are
"a sequence you often want individually". That was reasoning about saving files,
not about looking at the thing, and looking at all four passes at once is the
entire point of workflow 01. `sheet_layout` now covers TF Decode, TF Latent
Preview and TF Compare Levels too, and a test asserts *every* multi-frame node
offers it and defaults to stitched, so the next one cannot be forgotten.

**"The image generates but is not visible."** Three explanations, two wrong:

1. *Batch paging.* Right about the mechanism, wrong about the default --
   ComfyUI grids images until one is clicked. Corrected to the user.
2. *The ncat bridge is dropping the websocket.* Wrong, and the reasoning was
   seductive: both symptoms (no progress bar, no images) travel on the
   websocket, so one dead socket explains both, and the bridge was the one
   component nothing tested. `Ncat: Connection reset by peer` in every bridge
   log looked like proof. It is the ordinary teardown when the job ends.
3. Stale per-node `imageIndex` in a page that had never been reloaded. A hard
   reload fixed it.

The lesson is not "check harder" -- it is that a hypothesis explaining two
symptoms at once feels like evidence and is not. What settled it was building
the missing test rather than reading more code.

**`scripts/bridge_smoke.py`**, the fourth test layer. `server_smoke` talks to
ComfyUI on the compute node's own loopback, so it never crosses the bridge --
the exact path every human uses. The new script launches a real session and
drives workflow 01 through it as a browser would, checking that `progress`
events and the `executed` message carrying output images both arrive. 4/4, and
it is what proved hypothesis 2 wrong. `CONTRIBUTING.md` now tabulates what each
layer covers and what it is blind to; the honest line is that nothing here
drives a browser.

**One real fix fell out of it.** `ProgressBar` sends nothing when constructed --
its first message goes out on the first `update()` -- so the bar appeared only
*after* the checkpoint restore, the part that most needs it. It now sends 0/3 up
front.

**The clickable token grid works**, confirmed by use: it appears, typing syncs
it, and drag-paint works. Drag was broken on first delivery by
`setPointerCapture` on the pressed *cell*, which routes every later pointer
event to that element so siblings never receive `pointerenter` -- and would have
failed on touch too, where the browser captures to the pointerdown target
implicitly. It now captures on the board and hit-tests with
`document.elementFromPoint`.

## The region picker, finished (2026-09-03)

The original plan's step 2 was "16x16 grid overlay, click -> region-id lookup".
The grid landed yesterday; the region lookup did not, and the omission was not
cosmetic. `TF Tokens From Coords` snaps a selection to whole regions whenever a
map is wired (`min_overlap=0.0` -- a typed coordinate is a deliberate pick, not
a rough stroke), which is the default in workflows 02 and 05. So the widget
highlighted the one cell that was clicked while the node quietly took the forty
sharing its region, and the count beside the grid was wrong every time. A
half-built feature that lies is worse than an absent one.

The node now hands its map back to its own widget on a `tf_regions` key in the
`ui` payload, read via `onExecuted` -- the same public hook core's own AUDIO_UI
widget uses, chosen over `nodeOutputStore` because a file that has to keep
working should not lean on a store internal. Precedent for a custom `ui` key:
core's bbox editor reads `input_bboxes` back off its own node the same way.

One click now takes the whole region -- the paper's R_tgt, a semantic part
rather than an arbitrary token set -- and the grid draws the boundaries. The map
comes from the *previous* run, so a graph that has never run picks token-wise
and the node expands it, the same one-round-trip the Painter workflow already
has. Alt-click always picks the single token, because the `coords` text is what
a writeup quotes and it must be able to say exactly what was chosen.

`server_smoke` gained a criterion for the payload crossing the wire (16x16 ids,
50 regions at level 2). A browser is still the only thing that can prove the
widget *uses* it; that boundary is stated rather than blurred.

## Three from using the grid (2026-09-03)

**An empty selection crashed instead of stopping.** Clearing the grid let an
empty `TF_TOKENS` travel two nodes downstream and raise out of `TF Feature
Edit`, arriving as a raw traceback in a modal. The right shape already existed
-- `TF Tokens From Mask` stops the graph quietly with an instructive note -- and
`TF Tokens From Coords` now matches it: one `ExecutionBlocker` per declared
output plus `ui` text, never a message-carrying blocker (which renders as "Node
threw an error", and only on the first run). The downstream `require_nonempty`
guards stay: `TF Tokens Combine` can still hand over an empty selection by
intersecting two disjoint picks, and an edit must refuse that rather than write
nothing and report success. The tests that covered it were building their empty
selection *through* the coords node, so they were rewritten to construct one
directly -- otherwise they would have been testing the new block, not the guard.

**The visible level sliders ran to 15 when only 0-3 exist.** `level_input` used
`MAX_LEVELS - 1`, so three quarters of every level slider did nothing but get
silently clamped -- the same dead-knob shape as the old `which` + `level` pair,
which this repo has now paid for three times. Bounded to `SHIPPED_LEVELS - 1`,
because four levels is a property of the method rather than a per-run setting.
The advanced `-1 = auto` widgets keep the wider range deliberately: addressing a
deeper model than any released one is their entire documented purpose, and
behind the advanced toggle a wide range misleads nobody. Audited across every
node: five visible widgets now stop at 3, four advanced ones still reach 15.

**Alt is option on a Mac**, and the description of what it does was wrong in a
way worth correcting rather than quietly fixing. Alt-click changes the `coords`
*text*, not the selection: with a region map wired the node snaps at
`min_overlap=0.0`, so the whole region is selected either way. What it buys is
"7,7" in the field instead of a nine-run coordinate list -- the line a writeup
wants. Genuinely sub-region tokens need `regions` unwired. Stated that way now
in the node description, both READMEs and the widget's own tooltips.

## Nine outputs nothing could receive (2026-09-03)

Reported as "shift-drag from `max_distance` or `changed_tokens` suggests no
valid node, and the same on TF Save Report -- what are those outputs for?"

The suggestion list is not arbitrary. It comes from the frontend's
`extensions/core/slotDefaults.ts`, which skips every input whose type is in
`ComfyWidgets` (INT, FLOAT, STRING, BOOLEAN, COMBO) unless it is declared
`forceInput`. The one `forceInput` scalar in all of `comfy_extras` is
`floats_strength`, type `FLOATS`. So `LiteGraph.slot_types_default_out` has no
`"INT"` key whatsoever, and every INT/FLOAT drag dead-ends. `PreviewAny` is
indexed under `"*"` and never matches. STRING works only because
`TFSaveReport.text` is `force_input=True`.

Taken as a design test -- *an output socket is a promise something can receive
it* -- the outputs split on a line nobody had drawn on purpose. Of sixteen
scalar and string outputs, the five shipped workflows wired exactly three:
`TFImageNetClass.class_id`, `TFRegionMap.level` (six times) and
`TFSweep.report`. Nine of the rest were measurements: `changed_tokens`,
`max_distance`, `arms`, `spread`, `num_regions`, three `count`s, `num_levels`.
Nothing could ever receive one, because **you never drive a knob with a
measurement** -- and each was already in the node's own body and in the report
`TF Save Report` archives. All nine are gone; `TF Tokens Combine` traded its
`count` for an `info` string so the three selection nodes have one shape.

Two things this cost, both worth recording. `gpu_smoke` was a real consumer --
it asserted on `num_regions`, `arms` and `spread` directly. Two of those got
*better*: `regions.num_regions` rides on the object the `regions` socket already
carries. The spread did not: the per-arm finals it is computed from never leave
the node, so recomputing it independently would mean paying for three more
resumes, and the check now reads the node's own stated conclusion ("does change
the outcome") the way `every_arm_moved` already did. And three of the six cuts
shifted socket indices, `TFRegionMap.level` from 3 to 2 -- six links across the
workflows, cheap only because `make_workflows.py` regenerates them. That
generator addressed outputs by hard-coded index; resolving them by name was the
obvious follow-up, and is the section below.

`TestEveryScalarOutputDrivesAWidget` now holds the line: an INT/FLOAT output
must name the widget it feeds, and that widget is checked to exist with a
matching type. Verified to fail on the real bug by putting `changed_tokens`
back.

**And the smoke script went stale for the fourth time** -- a different way than
the three the AST test covers. `server_smoke.py` pinned `TFRegionMap`'s output
list as a literal `["TF_REGIONS", "IMAGE", "INT", "INT"]`, so job 449989 came
back 12/13 with every actual behaviour passing and only that expectation
failing. The AST check compares *call sites* against live schemas and cannot see
a hardcoded literal. Fixed by asserting what the check is for -- that the custom
socket types survive registration -- rather than the full arity, which the unit
suite now owns. The failure message was fixed at the same time: it printed the
success text next to `[FAIL]` whenever the types were wrong but no node was
missing, which is how a stale expectation reads as an unexplained failure.

Also corrected, in `CONTRIBUTING.md`, `sockets.node_preview` and a test
docstring: the claim that stock ComfyUI ships no node displaying a STRING,
"checked against every registered core class". `PreviewAny` does. The reason for
`node_preview` survives -- a result you have to bolt a second node onto is one
nobody reads -- but the sentence justifying it did not.

## The generator names its links (2026-09-04)

The follow-up left at the end of the section above, done. `make_workflows.py`
wired links as `(node_id, integer_slot)`; it now takes `(node_id, "output_name")`
and resolves the index from the schema, and an integer is **refused** rather than
accepted alongside a name -- allowing both is how the unstable form creeps back.

The argument for doing it is not the six hand-edits the `num_regions` cut cost.
It is that nothing would have caught getting one of them wrong.
`test_every_link_references_real_nodes_and_slots` compares the origin output's
type against the link's, and the link's type is taken from the *input* -- so it
separates a `TF_LEVELS` from an `IMAGE` and cannot separate two `INT`s. Until
`num_regions` was cut, `TFRegionMap` had two adjacent INT outputs, so wiring the
region count into `TFFeatureEdit.level` would have passed the entire suite and
produced a workflow that edits level 3 while claiming to edit level 2. That is
the silent-wrong-level failure `TokenSelection.level` exists to prevent, arriving
by a route the level check cannot see -- the level is *consistent*, it is just
the wrong one. The generator's own docstring already says positional drift is the
bug it exists to prevent; it was only preventing it for widget values, and
addressing inputs by id in the same function that counted to outputs.

**The refactor came with a proof rather than a test run.** All 84 links across
the five workflows were rewritten by hand, then the workflows regenerated: the
output is **byte-identical** to what the indices produced, so every name resolves
to the slot it replaced. A refactor of this shape either has that property or is
wrong, and checking it costs one `diff -r`.

`TestTheWorkflowGeneratorNamesTheOutputsItWires` holds the line, reading the
builders with `ast` the way `TestTheSmokeScriptsCallNodesCorrectly` reads
gpu_smoke's call sites. It needs no ComfyUI, because every link origin in these
workflows is one of this extension's own nodes -- `PreviewImage` has no outputs
and `Painter` is only ever an origin for `MASK`, which it names. Scoping the
variable map per builder function is not tidiness: `edit` is a TF Feature Edit in
workflow 02 and a TF Shape Edit in 04, and one map across the file would check
half the links against the wrong schema and pass.

Both layers were verified to fail on the real bug, not assumed to: reverting one
link to `(regions, 3)` fails `test_no_link_counts_to_a_slot` and raises
`TypeError` from the generator; naming the cut `num_regions` fails
`test_every_named_output_exists_on_the_node_it_comes_from` and raises `KeyError`.
352 tests, ruff clean. No GPU job was needed -- the generator runs on the login
node under `--cpu`, and nothing about node behaviour changed.

## The pictures are archived too (2026-09-04)

`Next` item 3 asked for the decoded intermediates to be saved beside the `.npz`.
**New: `TF Save Images`**, writing to `output/trajectory_forcing/<name>.png` under the same
`name` widget the other two save nodes use, so one run's `.npz`, `.md` and
`.png`s share a stem instead of landing under three unrelated names.

**The item's own framing was half wrong, and the correction is the useful part.**
It said a run that has to be cited "also wants the images that were looked at",
which treats every workflow alike. They are not alike:

- **A single trajectory** — the images are *re-derivable*. The `.npz` carries the
  latents, so `TF Load Levels` → `TF Decode` reproduces the frames exactly. It
  costs a GPU allocation and a model load; nothing is lost.
- **A sweep** — the frames are *not recoverable at all*. Only `output_arm`'s
  trajectory leaves the node on the `levels` socket; every other arm's latents
  are discarded when `execute` returns. Those arms exist as pixels in the contact
  sheet and nowhere else, and the sheet went to `PreviewImage`, which writes to
  ComfyUI's **temp** directory. Re-running with the same seeds would rebuild
  them, but that is redoing the experiment rather than reading its record.

So the item was right that this was more pressing once the sweep existed, and
wrong about why: not because a sheet is "worth keeping" but because it is the
only artefact here that a cache clear destroys outright. Workflow 05 is the one
that got the node wired in.

**Core's `SaveImage` was the alternative and was not dismissed lightly** — the
Painter pivot is this repo's own precedent for using what core ships rather than
rebuilding it. It writes PNGs perfectly well. What it cannot do is tie the files
to the run: `ComfyUI_00001_.png` in the output root, while the trajectory and the
table sit elsewhere under their own names. The shared stem is the whole feature,
and the class/seed/history stamped into each PNG's own metadata is what makes a
picture in a writeup traceable back to the run that made it.

Two smaller things came with it. The empty-batch case is a `ValueError` naming
the shape rather than an `IndexError` from PIL, because a degenerate result is a
result and this repo has already lost a run to one reaching a library that
raises. And not wiring the trajectory in is reported in the node's own body
("no class/seed metadata"), since silently writing untraceable files is the exact
failure the node exists to prevent.

**`server_smoke` gained criterion 6**, written into its docstring before the run:
05 leaves both its table and its sheet on disk under one name, and the PNG
carries the class and seed that produced it. The directory is cleared before the
workflows run — `TF Save Report` appends and `TF Save Images` side-steps a
collision with `-001`, so both are designed *not* to overwrite, which means a
file from an earlier run would have satisfied the criterion while this run wrote
nothing. A criterion an absent result can pass has reported PASS here before.

**And a sixth staleness vector in that script, closed before it bit.**
`EXPECTED_NODES` is a hardcoded set and the check computes `EXPECTED_NODES -
published`, so a node added to the extension and forgotten there does not fail —
it silently stops being covered. That is the hardcoded-output-list bug of job
449989 in a quieter form: under-checking never turns red at all.
`test_server_smoke_expects_exactly_the_nodes_that_exist` now pins the set equal
to the registered node ids, in the unit suite.

## Next

Ordered by what would change the design, not by size.

1. **Keep using it.** Everything below is still a guess until a specific edit has
   been made end to end, and then swept.
2. **Region granularity.** Cosine threshold 0.9 gives ~50 regions over 256 tokens
   at level 2 — subpart-scale, which is right for the level but fiddly to paint
   against. Worth checking whether a per-level default (looser at level 0,
   tighter at level 3) makes the region map more useful than one slider does.
   The sweep can now answer this rather than it staying a guess: `cosine_threshold`
   is not an axis, but running one edit at four thresholds by hand is four runs
   of workflow 05, and the spread numbers are comparable across them.
3. ~~**Save the decoded intermediates alongside the `.npz`.**~~ Done — see "The
   pictures are archived too" below. The item's framing was half wrong, which is
   recorded there.
4. **A second sweep axis.** A 4×4 grid (seed × l\*) is what an actual results
   table wants, and the repo's own experiment notes are emphatic that a
   four-seed cell has already overstated a gap here badly enough to need a
   public correction. One axis is the honest starting point — it keeps "one
   variable per arm" true by construction — but the cell size argument points
   at two.
