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

## Status — 2026-09-03

Steps 1–5 of the original build order are done and verified on an H100; the
open questions below are all answered. What has *not* happened is a session of
someone using it for a real edit, which is the next thing.

| build step | state |
|---|---|
| 1. Load + Generate + Decode | done — `gpu_smoke` 7/7, job 449604 |
| 2. Region picker, read-only | done, but not as a custom widget — see the pivot below |
| 3. Feature edit (same-canvas and cross-canvas) | done — `TF Feature Edit`, both source modes |
| 4. Resume, closing the loop | done — `TF Resume From Level` |
| 5. Shape edit | done — `TF Shape Edit`, region-mean reassignment |

Verification, all reproducible from this repo:

- `pytest tests` — 158 tests, no GPU, ~5 s. Edit math, payloads, drawing, and
  every node's schema and `execute` against a stub pipeline.
- `slurm/gpu_smoke.sbatch` — the nodes against the real TF-L model. **7/7,
  job 449604 on mlcbm011**, including the control (same class + seed reproduces
  the trajectory bit-for-bit). Load 14.6 s warm, generate 0.1 s, resume 2.2 s.
- `slurm/server_smoke.sbatch` — the workflows through a real ComfyUI server.
  **9/9, job 449655 on mlcbm011.** The first attempt (449598) was 2/6 and is
  what found both the namespace bug below and a workflow naming an ImageNet
  class string one word off. The criteria covering the painter
  workflow's quiet stop were added after the first browser session and took five
  runs to get right (449615, 449616, 449617, 449618, 449655) — every failure was
  a real defect, and the last two only became visible once the test ran the
  workflow twice and watched the websocket.

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
`~/.ssh/id_ed25519.pub`). Replaced with `./serve.sh`: `srun` in the foreground so
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

**`serve.sh` said too little at the start and too much at the end.** It now
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
`workflows/README.md` (a tutorial for the four examples) and `CONTRIBUTING.md`
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
column numbers on every grid preview), `TF Tokens From Coords` works without a
`levels` wire, and `TF Levels Info` reports the class *name*.

**New: `TF Compare Levels`.** Two trajectories in, tokens-changed per level and
a heatmap of where out. The question every edit prompts is "did that do
anything", and the only previous answer was two pictures and your judgement. An
edit at l* should show zero change below l*, the selected tokens at l*, and
diffuse change above -- anything else is the interesting kind of wrong, and now
visible as a number.

Not done, and still the biggest gap: **a sweep node**. Running the same edit
across seeds or across l* is the thing a graph should make cheap and the Gradio
app cannot, and it still means duplicating the graph by hand.

## Next

Ordered by what would change the design, not by size.

1. **Keep using it.** The two fixes above came from one launch. Everything below
   is still a guess until a specific edit has been made end to end.
2. **Region granularity.** Cosine threshold 0.9 gives ~50 regions over 256 tokens
   at level 2 — subpart-scale, which is right for the level but fiddly to paint
   against. Worth checking whether a per-level default (looser at level 0,
   tighter at level 3) makes the region map more useful than one slider does.
3. **A batch/sweep node.** The thing this graph makes cheap that the Gradio app
   does not is running the same edit across seeds or across *l\**. A node that
   fans a trajectory out over a seed list would turn the extension into an
   experiment tool rather than a demo.
4. **Save the decoded intermediates alongside the `.npz`.** `TF Save Levels`
   stores latents and history; a run that has to be cited later also wants the
   images that were looked at.
