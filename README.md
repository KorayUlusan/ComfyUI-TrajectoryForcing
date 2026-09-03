# ComfyUI-TrajectoryForcing

A ComfyUI custom-node extension for [Trajectory Forcing](https://mervekocabas.github.io/TrajectoryForcing/)
(TF, ECCV 2026): coarse-to-fine generation with a decodable preview at every
level, and the paper's interactive latent-token editing, as a node graph.

TF generates an image by walking a learned trajectory through a hierarchical
DINOv2 latent space -- object/background, then parts, then subparts, then the
finest tokens -- one network evaluation per level. Because the RAE decoder is
frozen and works on any point in that space, *every* intermediate level decodes
to a picture, which is what makes the trajectory something a user can inspect,
edit, and resume from. That structure is a DAG with visible intermediates and a
feedback edge, which is a better fit for a node graph than for a single page.

**Status: working end to end, not yet used in anger.** Every node runs against
the real TF-L model on an H100, the example workflows execute through a real
ComfyUI server, and the tests below pass. What it has not had is a session of
someone actually trying to make a specific edit and finding out where the
interaction model gets in the way.

```
[TF Load Pipeline] ─┬─► [TF Generate] ──┬─► [TF Decode Levels] ──► [Preview Image]
                    │                   ├─► [TF Latent Preview] ─► [Preview Image]
                    │                   │
                    │                   ├─► [TF Region Map] ──┐
                    │                   │                     │
                    │                   └─► [TF Level Canvas] ┴─► [Painter] ─► [TF Tokens From Mask]
                    │                                                                    │
                    │                            [TF Feature Edit] / [TF Shape Edit] ◄───┘
                    │                                    │
                    └────────────────────► [TF Resume From Level] ──► [TF Decode Levels]
```

## Install

Needs a GPU. TF runs a JAX flow model and a PyTorch RAE decoder in the same
process, so one environment has to hold both stacks:

```bash
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI      # if you have none
ln -s /path/to/ComfyUI-TrajectoryForcing ~/ComfyUI/custom_nodes/

bash env/setup.sh          # builds ~/.venvs/comfyui-tf (~10 min, ~11 GB)
./run_comfyui.sh           # http://localhost:8188
```

`env/setup.sh` documents the install order and the one pin that had to move
(ComfyUI 0.34 needs a newer torch than TrajectoryForcing pins); see
`requirements.txt` for the reasoning.

The extension needs a **TrajectoryForcing checkout** to import the model from --
it is never vendored, so the code here always tracks the pinned submodule rather
than a stale copy of its math. It is found as a sibling directory, or set
`TF_REPO`. Weights are resolved on first use:

| what | where | how |
|---|---|---|
| flow checkpoint | `ComfyUI/models/trajectory_forcing/` | dropdown; `auto` downloads `TF_L_edit` into it |
| RAE decoder | `TrajectoryForcing/checkpoints/rae/` | reused if present, else downloaded; override with `TF_RAE_ROOT` |

On a Slurm cluster, `sbatch slurm/comfyui.sbatch` runs the server on a GPU node
and prints the SSH port-forward to reach it.

## The nodes

All under the **TrajectoryForcing** category.

### Generating and looking

| node | what it does |
|---|---|
| **TF Load Pipeline** | Loads the flow model and RAE decoder. Cached for the life of the process, so re-queueing never re-reads the checkpoint. `warmup` pays the XLA compile here instead of on your first prompt. |
| **TF ImageNet Class** | Pick a class by name, get its id. |
| **TF Generate** | Samples one full trajectory. Outputs every level, not just the last. |
| **TF Decode Levels** | RAE-decodes all levels, the final one, or a single one. |
| **TF Latent Preview (PCA)** | The token grid as PCA false colour -- far cheaper than decoding, and it shows the structure the edits act on. `palette_from` fits the colours jointly with a second trajectory so two images are comparable. |
| **TF Levels Info** | Shape, class, seed, and the edit history of a trajectory. |

### Choosing what to edit

TF's edits operate on regions of a 16x16 token grid, which is not a resolution
any ComfyUI mask tool knows about. Rather than a bespoke canvas widget, **TF
Level Canvas** renders a level as a paintable image with the token grid drawn
on, core **Painter** supplies the brush, and **TF Tokens From Mask** converts
what was painted back down to tokens.

| node | what it does |
|---|---|
| **TF Level Canvas** | One level as a 512px canvas: PCA latent or decoded RGB, with the token grid, region boundaries, and an existing selection drawn on. Feed it to Painter. |
| **TF Region Map** | Clusters a level's tokens into connected regions by cosine similarity. These are the *R* in the paper's edits. The threshold sets granularity -- 0.9 gave ~50 regions over 256 tokens at level 2. |
| **TF Tokens From Mask** | Painted mask → token selection. A token counts once enough of its footprint is painted, so a stroke clipping a corner does not overwrite that token's whole feature vector. Wire a region map in to snap a rough stroke to whole regions. |
| **TF Tokens From Coords** | Type `row,col` pairs (`7,6:9` is a run). Reproducible in a way a brush stroke is not, which is what a written-up experiment needs. |
| **TF Tokens Combine** | Union / intersection / difference / invert, to build a region up from several strokes. |
| **TF Tokens Preview** | Draw a selection on its own, to check what a mask resolved to. |

### Editing and resuming

Both edits reduce to the paper's one primitive -- `z̃ᵢ = f_src` for the selected
tokens -- and differ only in where `f_src` comes from. Neither samples anything;
they produce the edited canvas at level *l\**, and **TF Resume From Level** is
what propagates it.

| node | what it does |
|---|---|
| **TF Feature Edit** | Replaces the target tokens' features with one sourced from elsewhere -- same trajectory or a second one. `region mean` is the paper's `f_src` (one averaged vector fills the target); `token cycle` copies token-for-token. `strength` interpolates rather than replacing. |
| **TF Shape Edit** | Hands boundary tokens from one region to a neighbour: they take on the *receiving region's* mean feature, so its extent changes and its content does not. |
| **TF Resume From Level** | Re-samples every level above *l\**, conditioned on the canvas sitting there. Levels below are untouched -- sampling is Markov in the level index, so an edit only ever propagates upward. `follow_edit` picks up *l\** from the upstream edit node. |
| **TF Save / Load Levels** | A trajectory to and from `output/trajectory_forcing/*.npz`, with its class, seed and edit history. A trajectory costs GPU time and is what every edit is measured against; reloading the exact one an earlier run used is what makes two edits comparable. |

Coarser edits (small *l\**) cascade through more levels and have broad semantic
impact; finer edits stay spatially local. A stack that has been edited but not
resumed is marked, and **TF Decode Levels** warns rather than silently showing
you the pre-edit trajectory above *l\**.

## Example workflows

`workflows/` holds four, generated from the live node schemas by
`scripts/make_workflows.py` rather than written by hand -- LiteGraph stores
widget values positionally, so a hand-written file drifts silently when a widget
is added. Each is emitted twice: LiteGraph format for the **Open** menu, and API
format under `workflows/api/` for `POST /prompt`.

| workflow | |
|---|---|
| `01-generate-and-decode` | The vertical slice: one trajectory, every level, decoded and as PCA. |
| `02-feature-edit-coords` | Cross-image feature edit driven by typed coordinates. Runs as-is, no painting; the reproducible version. |
| `03-feature-edit-painter` | The same edit, painted. Run once to get the canvas, paint on the Painter node, run again. |
| `04-shape-edit` | Moving a region boundary instead of a feature. |

## Tests

```bash
pytest tests                      # 132 tests, no GPU, ~4 s
sbatch slurm/gpu_smoke.sbatch     # the nodes against the real model
sbatch slurm/server_smoke.sbatch  # the workflows through a real ComfyUI server
```

The unit tests cover the edit math, the payload types, the drawing, and every
node's schema and `execute` against a stub pipeline -- including that each
schema input matches the `execute` parameter it feeds, which is otherwise
invisible until someone queues the node in a browser.

The two Slurm jobs exist because they catch different things, and both caught
real bugs that the unit tests could not:

* `gpu_smoke` runs the nodes directly against the real TF-L model. Its exit
  criteria are in the script's docstring and include a control -- the same class
  and seed must reproduce the trajectory bit-for-bit, or the run is void.
* `server_smoke` starts a real ComfyUI server and executes the workflows through
  it. This is what covers ComfyUI's own validation of the custom socket types,
  and whether the generated workflows are executable rather than merely
  well-formed. It found the import bug described below.

## Two things worth knowing if you touch this

**TrajectoryForcing and ComfyUI both own a top-level `utils`.** TF is a flat
repo whose code imports `utils`, `models`, `configs` and `third_party` by those
bare names; ComfyUI ships its own `utils` package. Putting TF's root on
`sys.path` is not enough and is actively harmful in both directions, so
`tf_nodes/tf_import.py` swaps the two namespaces instead of merging them. Every
call into TF has to happen inside `tf_scope()` -- not just the import -- because
the RAE decoder imports `utils.logging_util` and `third_party.rae_decoder`
lazily, from inside functions that first run at decode time.

**None of TF's directories has an `__init__.py`**, so they are all PEP 420
namespace packages, and CPython gives a namespace package `__file__ = None`
rather than no `__file__`. `inspect.getmodule` walks `sys.modules` guarded by
`hasattr(m, "__file__")`, passes that guard, and dies in `inspect.getfile` --
but only once the name has previously been recorded with a real path, which is
exactly what the namespace swap arranges. pydantic calls `getmodule` while
building a model class and wandb builds pydantic models while TF imports it, so
this crashed TF Load Pipeline inside a running server while the identical import
succeeded in a bare script. `tf_import.py` binds every namespace directory in
TF's tree up front with the null `__file__` removed;
`tests/test_tf_import.py::test_a_swapped_module_does_not_break_a_sys_modules_walk`
reproduces the crash.

## JAX and PyTorch on one card

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is the important setting: at its default
JAX claims ~75% of the card the moment it initialises, and
`comfy.model_management` cannot see that allocation, so it keeps loading torch
models into VRAM that is already gone. `run_comfyui.sh` sets it before ComfyUI
starts; the extension sets the same values at load as a fallback, so ComfyUI
started any other way still works, and anything already exported wins.

A hard ceiling on JAX's share is available but off by default: `MEM_FRACTION`
only sizes the *preallocated* block, so growth-on-demand and a cap are
alternatives rather than a pair. `TF_XLA_MEM_FRACTION=0.3 ./run_comfyui.sh`
flips both and passes ComfyUI a matching `--reserve-vram`. TF-L plus the ViT-XL
decoder is a few GB, so the cap is only worth it in a workflow that also loads
something big.

## Not here

**TF-2.0 (text conditioning)** has no trained checkpoints, so a `TF Text Encode`
node would have nothing to load against. **Trajectory Dreamer** (the 3D work) is
a separate project and is not touched here.

## Citation

```bib
@Inproceedings{kocabas2026trajectoryforcing,
  author    = {Kocabas, Merve and Gao, Gege and Schölkopf, Bernhard and Geiger, Andreas},
  title     = {Trajectory Forcing: Structure-First Generation with Controllable Semantic Trajectories},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
