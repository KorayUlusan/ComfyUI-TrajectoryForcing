# ComfyUI-TrajectoryForcing

ComfyUI nodes for [Trajectory Forcing](https://mervekocabas.github.io/TrajectoryForcing/)
(TF, ECCV 2026): coarse-to-fine image generation with a decodable preview at
every level, and the paper's interactive latent-token editing, as a node graph.

> **First time with ComfyUI?** Start with
> **[docs/GETTING-STARTED.md](docs/GETTING-STARTED.md)** — it assumes nothing and
> gets you from an empty terminal to your first edit.

![four levels of one trajectory](docs/img/trajectory.png)

TF builds an image in four passes through a hierarchical DINOv2 latent space —
object/background, then parts, then subparts, then the finest tokens — one
network evaluation each. The RAE decoder is frozen and works on any point in
that space, so *every* intermediate level decodes to a picture. That is what
makes the trajectory something you can inspect, edit and resume from, and it is
a DAG with visible intermediates and a feedback edge — a better fit for a node
graph than for a single page.

![before and after an edit](docs/img/edit.png)

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA, **8 GB VRAM minimum**, 12 GB comfortable. No CPU or Apple Silicon path. |
| OS | Linux (developed on RHEL 9) or Windows with CUDA |
| Disk | ~11 GB environment + ~13 GB model weights |
| Python | 3.11 |

VRAM figures are measured rather than estimated — `scripts/measure_resources.py`
produces them, and on an H100 the peak for the full editing workflow was
**6.6 GiB**, with 5.3 GiB of host RAM:

| stage | VRAM |
|---|---|
| model loaded | 2.5 GiB |
| + first generate (compiles XLA) | 4.6 GiB |
| + RAE decoder built | 6.6 GiB |
| full edit, two trajectories held | 6.6 GiB |

That is TrajectoryForcing's share only. A workflow that also loads an SD
checkpoint needs room for both, so on an 8 GB card keep this the only model in
the graph.

---

## Install

Both routes need a **TrajectoryForcing checkout** — the model code is imported
from it rather than vendored, so this extension always tracks the pinned
upstream rather than a stale copy of its math. Point `TF_REPO` at it, or put it
next to this directory.

### Option A: your own machine

```bash
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
git clone https://github.com/mervekocabas/TrajectoryForcing ~/TrajectoryForcing
git clone <this-repo> ~/ComfyUI-TrajectoryForcing
ln -s ~/ComfyUI-TrajectoryForcing ~/ComfyUI/custom_nodes/ComfyUI-TrajectoryForcing

cd ~/ComfyUI-TrajectoryForcing
export TF_REPO=~/TrajectoryForcing
bash env/setup.sh          # ~10 min, ~11 GB
./run_comfyui.sh           # then open http://localhost:8188
```

Wait for `To see the GUI go to: http://0.0.0.0:8188` — the port is not open
before that line appears.

`env/setup.sh` documents the install order and the one pin that had to move:
ComfyUI 0.34's `comfy-kitchen` needs a newer torch than TrajectoryForcing pins,
so this runs torch 2.8.0+cu128. `requirements.txt` says why that is safe here and
where it would not be.

### Option B: a Slurm cluster

```bash
./serve.sh                 # or ./serve.sh 8188
```

Allocates a GPU node, streams the log to your terminal, and **Ctrl-C cancels the
job** — `srun`, not `sbatch`, so quitting does not leave a GPU allocated until
the walltime expires. The trade is that the job dies with the terminal, which is
right for an interactive session and wrong for anything else.

ComfyUI binds the *compute* node, which your laptop cannot route to, so
`serve.sh` also bridges the login node's `localhost:PORT` to it with `ncat`.
That avoids an SSH hop from login to compute, which needs a password unless your
`~/.ssh/authorized_keys` contains your own `~/.ssh/id_*.pub` (`$HOME` is shared
across nodes, so appending it once fixes that route too). From your laptop:

```bash
ssh -N -L 8188:localhost:8188 <user>@<login-node>
```

or nothing at all under VS Code Remote, which forwards `localhost` ports itself.

Tune with `TF_PARTITION`, `TF_QOS`, `TF_TIME`.

### Weights

Both routes resolve these on first use — nothing to download by hand:

| what | where | how |
|---|---|---|
| flow checkpoint | `ComfyUI/models/trajectory_forcing/` | dropdown; `auto` downloads `TF_L_edit` into it |
| RAE decoder | `TrajectoryForcing/checkpoints/rae/` | reused if present, else downloaded; override with `TF_RAE_ROOT` |

---

## Start here

Four example workflows, walked through in
**[workflows/README.md](workflows/README.md)** and self-documenting on the
canvas — each opens with a note explaining it, with the nodes boxed into
numbered groups.

| | | runs as-is? |
|---|---|---|
| `01-generate-and-decode` | the method, no editing | yes |
| `02-feature-edit-coords` | change a region's content | yes |
| `03-feature-edit-painter` | the same, with a brush | needs two runs |
| `04-shape-edit` | move a region's boundary | yes |

---

## The nodes

All under the **TrajectoryForcing** category.

### Generating and looking

| node | what it does |
|---|---|
| **TF Load Pipeline** | Loads the flow model and RAE decoder. Cached for the life of the process, so re-queueing never re-reads the checkpoint. `warmup` pays the XLA compile here instead of on your first prompt. |
| **TF ImageNet Class** | Pick a class by name, get its id. |
| **TF Generate** | Samples one full trajectory. Outputs every level, not just the last. |
| **TF Decode Levels** | RAE-decodes all levels, the final one, or a single one. |
| **TF Latent Preview (PCA)** | The token grid as PCA false colour — far cheaper than decoding, and it shows the structure the edits act on. `palette_from` fits the colours jointly with a second trajectory so two images are comparable. |
| **TF Levels Info** | Shape, class id and name, seed, and the edit history of a trajectory. |
| **TF Compare Levels** | What changed between two trajectories, per level and per token: a report, and a heatmap of where. Answers "did the edit do anything" with a number instead of eyeballs — an edit at *l\** should leave every level below it untouched. |

### Choosing what to edit

TF's edits operate on regions of a 16×16 token grid, which is not a resolution
any ComfyUI mask tool knows about. Rather than a bespoke canvas widget, **TF
Level Canvas** renders a level as a paintable image with the token grid drawn
on, core **Painter** supplies the brush, and **TF Tokens From Mask** converts
what was painted back down to tokens.

| node | what it does |
|---|---|
| **TF Level Canvas** | One level as a 512px canvas: PCA latent or decoded RGB, with the token grid, region boundaries, and an existing selection drawn on. Feed it to Painter. |
| **TF Region Map** | Clusters a level's tokens into connected regions by cosine similarity. These are the *R* in the paper's edits. The threshold sets granularity — 0.9 gives ~50 regions over 256 tokens at level 2. Its `level` output is worth wiring into the edit node, so the two cannot drift apart. |
| **TF Tokens From Mask** | Painted mask → token selection. A token counts once enough of its footprint is painted, so a stroke clipping a corner does not overwrite that token's whole feature vector. Wire a region map in to snap a rough stroke to whole regions. |
| **TF Tokens From Coords** | Type `row,col` pairs (`7,6:9` is a run). Reproducible in a way a brush stroke is not, which is what a written-up experiment needs. |
| **TF Tokens Combine** | Union / intersection / difference / invert, to build a region up from several strokes. |
| **TF Tokens Preview** | Draw a selection on its own, to check what a mask resolved to. |

### Editing and resuming

Both edits reduce to the paper's one primitive — `z̃ᵢ = f_src` for the selected
tokens — and differ only in where `f_src` comes from. Neither samples anything;
they produce the edited canvas at level *l\**, and **TF Resume From Level** is
what propagates it.

| node | what it does |
|---|---|
| **TF Feature Edit** | Replaces the target tokens' features with one sourced from elsewhere — same trajectory or a second one. `region mean` is the paper's `f_src` (one averaged vector fills the target); `token cycle` copies token-for-token. `strength` interpolates rather than replacing. |
| **TF Shape Edit** | Hands boundary tokens from one region to a neighbour: they take on the *receiving region's* mean feature, so its extent changes and its content does not. |
| **TF Resume From Level** | Re-samples every level above *l\**, conditioned on the canvas sitting there. Levels below are untouched — sampling is Markov in the level index, so an edit only ever propagates upward. `follow_edit` picks up *l\** from the upstream edit node. |
| **TF Save / Load Levels** | A trajectory to and from `output/trajectory_forcing/*.npz`, with its class, seed and edit history. A trajectory costs GPU time and is what every edit is measured against; reloading the exact one an earlier run used is what makes two edits comparable. |

**Most nodes need no `pipeline` wire.** A trajectory carries the pipeline that
produced it, so TF Decode, TF Latent Preview, TF Level Canvas, TF Resume and TF
Compare all find it on their own. The socket is there as an override, and for a
trajectory restored by TF Load Levels, which has none.

**A selection remembers which level's regions it was snapped to,** and the edit
nodes refuse to apply it at a different one. Every level shares the same token
grid, so without that check the edit lands on the wrong region and nothing says
so.

Coarser edits (small *l\**) cascade through more levels and have broad semantic
impact; finer edits stay spatially local. A stack that has been edited but not
resumed is marked, and **TF Decode Levels** warns rather than silently showing
you the pre-edit trajectory above *l\**.

---

## Troubleshooting

**"Nothing painted yet" in workflow 03.** Working as intended — that workflow
needs two runs. See
[workflows/README.md](workflows/README.md#03--feature-edit-painter).

**The first run takes minutes.** Loading the checkpoint plus an XLA compile.
Once per ComfyUI start; later runs take about a second. `warmup` on *TF Load
Pipeline* moves the cost to load time so it does not look like a hung prompt.

**`Could not find the TrajectoryForcing checkout`.** Set `TF_REPO`, or put the
checkout next to this directory.

**`ERROR: CUDA is not available`.** The GPU is not visible to Python. Check
`nvidia-smi`, and that the venv was built on the same machine.

**Out of memory with another model in the graph.** JAX and torch share the card,
and `comfy.model_management` cannot see JAX's allocations. Set
`TF_XLA_MEM_FRACTION=0.3` to cap JAX and pass ComfyUI a matching
`--reserve-vram`; the note in `run_comfyui.sh` explains why a cap and
grow-on-demand are alternatives rather than a pair.

**An edit appears to do nothing.** Wire *TF Compare Levels* to the before and
after trajectories: it reports tokens changed per level. If the answer is zero,
check the *what was selected* preview — most often the selection is not where
you thought. If the levels above your edit look stale, *TF Resume From Level*
has not run.

**"built from level N's regions but is being applied at level M".** A selection
was snapped to a region map from a different level. Point *TF Region Map* at the
level you are editing, or wire its `level` output into the edit node.

---

## Not here

**TF-2.0 (text conditioning)** has no trained checkpoints, so a `TF Text Encode`
node would have nothing to load against. **Trajectory Dreamer** (the 3D work) is
a separate project.

---

## Contributing

Architecture, tests, and the sharp edges worth knowing before changing anything:
**[CONTRIBUTING.md](CONTRIBUTING.md)**. Design history and the reasoning behind
each pivot: **[PLAN.md](PLAN.md)**.

## Citation

```bib
@Inproceedings{kocabas2026trajectoryforcing,
  author    = {Kocabas, Merve and Gao, Gege and Schölkopf, Bernhard and Geiger, Andreas},
  title     = {Trajectory Forcing: Structure-First Generation with Controllable Semantic Trajectories},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
