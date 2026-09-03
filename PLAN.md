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

- `pytest tests` — 132 tests, no GPU, ~4 s. Edit math, payloads, drawing, and
  every node's schema and `execute` against a stub pipeline.
- `slurm/gpu_smoke.sbatch` — the nodes against the real TF-L model. **7/7,
  job 449604 on mlcbm011**, including the control (same class + seed reproduces
  the trajectory bit-for-bit). Load 14.6 s warm, generate 0.1 s, resume 2.2 s.
- `slurm/server_smoke.sbatch` — the workflows through a real ComfyUI server.
  **6/6, job 449601 on mlcbm009.** The first attempt (449598) was 2/6 and is
  what found both the namespace bug below and a workflow naming an ImageNet
  class string one word off; 449599 was the first clean pass, 449601 re-ran it
  after the last round of edits.

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

## Next

Ordered by what would change the design, not by size.

1. **Use it.** Make one specific edit end to end in the browser and write down
   where the interaction gets in the way. Everything below is a guess until that
   happens.
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
