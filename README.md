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
./run_comfyui_slurm.sh                 # or ./run_comfyui_slurm.sh 8188
```

Allocates a GPU node, streams the log to your terminal, and **Ctrl-C cancels the
job** — `srun`, not `sbatch`, so quitting does not leave a GPU allocated until
the walltime expires. The trade is that the job dies with the terminal, which is
right for an interactive session and wrong for anything else.

ComfyUI binds the *compute* node, which your laptop cannot route to, so
`run_comfyui_slurm.sh` also bridges the login node's `localhost:PORT` to it with `ncat`.
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

Five example workflows, walked through in
**[workflows/README.md](workflows/README.md)** and self-documenting on the
canvas — each opens with a note explaining it, with the nodes boxed into
numbered groups.

| | | runs as-is? |
|---|---|---|
| `01-generate-and-decode` | the method, no editing | yes |
| `02-feature-edit-coords` | change a region's content | yes |
| `03-feature-edit-painter` | the same, with a brush | needs two runs |
| `04-shape-edit` | move a region's boundary | yes |
| `05-sweep-seeds` | one edit across four seeds, tabulated | yes |

---

## The nodes

Under **TrajectoryForcing** in the node menu, split into `generate`, `select`,
`edit` and `save and load`. Every node is searchable by what it does as well as
its name — "paint", "mask", "diff", "region", "seed" all find the right one.

**`-1` always means "decide for me".** A handful of advanced widgets take it,
and each one's label says so — `level (-1 = auto)`, `class id (-1 = auto)`,
`source level (-1 = auto)`. The node then reports what auto chose, so you never
have to infer it:

```
resume from level 2 (auto: the level the edit wrote to); class 213 (auto: the
trajectory's own); seed 592
levels 3..3 re-sampled
```

Set a real value and it is used instead, reported as *set on the node*.

**Rarely-used settings are hidden behind ComfyUI's advanced toggle** rather
than removed — about half of them. Turn on *Settings → Always show advanced
widgets* to see everything a node can do.

**Every node shows its own result in its body**: the edit summary, the region
count, the selection, the comparison table. Nothing needs wiring to a preview
node to be read.

**Measurements are text, not sockets.** The tokens-changed total, the peak
distance, the spread across a sweep's arms, the region and selection counts —
all of them are in the node's own body and in the `report` string, which *TF
Save Report* writes to a file. None of them is an output you can drag from,
because ComfyUI has nowhere to put a number: its suggestion index skips every
INT and FLOAT input that is an ordinary widget, so a drag from one dead-ends in
an empty menu. Wire `report` into *TF Save Report* to keep the numbers; the
outputs that *are* sockets — a region map's `level`, a class id, a seed — are
there because a widget on another node receives them.

**Several frames arrive as one image.** ComfyUI shows a multi-image output one
frame at a time, behind a small `1/4` button that is easy to miss — so four
decoded levels would look like level 0 alone. Any node that emits frames meant
to be seen together stitches them into a single contact sheet instead: a row up
to six, a near-square grid beyond. `sheet_layout` (advanced) switches back to a
batch, which is the only shape *SaveImage* writes as one file per frame.

### Generating and looking

| node | what it does |
|---|---|
| **TF Load Pipeline** | Loads the flow model and RAE decoder, with a progress bar across restore → compile → decoder build, and a `ready in Ns` when it lands. Cached for the life of the process, so re-queueing never re-reads the checkpoint. Leave `warmup` on: turning it off does not save the 1–2 minutes, it moves them to your first TF Generate, which cannot show progress for a single opaque compile. |
| **TF ImageNet Class** | Pick a class by name, get its id. |
| **TF Generate** | Samples one full trajectory. Outputs every level, not just the last. |
| **TF Decode Levels** | RAE-decodes all levels, the final one, or one you name — a single dropdown, not a mode plus a number. |
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
| **TF Tokens From Coords** | **Click a 16×16 grid on the node**, or type `row,col` pairs (`7,6:9` is a run) — the grid writes into the text field, so both are the same value and either way the selection stays a text string you can paste into a writeup. Wire a region map in and one click takes the whole region, with the boundaries drawn on the grid; alt-click (option on a Mac) writes just that one coordinate, which is what a writeup wants — though with a map wired the node still snaps to the whole region either way. |
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
| **TF Resume From Level** | Re-samples every level above *l\**, conditioned on the canvas sitting there. Levels below are untouched — sampling is Markov in the level index, so an edit only ever propagates upward. Left alone it follows the upstream edit, so *l\** cannot disagree with where the edit landed. |
| **TF Sweep Edit** | The whole chain above, run once per value of **one** axis — a seed list, the levels, or a strength ramp — with everything else pinned. See below. |
| **TF Save / Load Levels** | A trajectory to and from `output/trajectory_forcing/*.npz`, with its class, seed and edit history. A trajectory costs GPU time and is what every edit is measured against; reloading the exact one an earlier run used is what makes two edits comparable. |
| **TF Save Report** | A sweep or comparison table to `output/trajectory_forcing/<name>.md`, fenced so the columns survive, and stamped with the class, seed and edit history that produced it when a trajectory is wired in. Appends by default, so a session's runs accumulate into one file. Without it the numbers exist only as text in a node body. |
| **TF Save Images** | Pictures to `output/trajectory_forcing/<name>.png`, beside the `.npz` and the `.md` from the same run, with the class, seed and edit history stamped into each PNG's metadata. Every workflow otherwise ends in `PreviewImage`, which writes to ComfyUI's *temp* directory. For one trajectory that is recoverable — the `.npz` has the latents, so decoding rebuilds the frames — but a sweep's non-output arms exist in the contact sheet and nowhere else. |

### Sweeping

The reason to prefer a graph over the editing env's Gradio app is that the same
edit can be run across a seed list, or across *l\**, and tabulated. **TF Sweep
Edit** is that loop: feature edit → resume → measure, once per value, with
everything not on the axis pinned to its own widget, so no two arms ever differ
in two ways at once.

Two things make it an experiment tool rather than a batch button:

- **Every arm is measured against its own baseline** — the same trajectory
  resumed from the same level with the same seed and *no* edit. Re-sampling
  from an edited canvas is still sampling, so without that, "the final image
  changed a lot" cannot be told apart from "the resume seed changed a lot", and
  a seed sweep mostly measures the seed. (`baseline` is advanced and can be
  turned off; the report says which reference it used either way.)
- **The spread across arms is reported** — the mean pairwise cosine distance
  between arms at the final level. Near zero means every arm landed in the same
  place, so the *edit* decides the outcome and a single-seed result was
  trustworthy. Large means the seed does, and it was not.

The node's body carries the table and the contact sheet — the no-edit baseline
first, then one frame per arm, stitched into **one** image so the arms can be
compared at full size (a row up to six of them, a near-square grid beyond). Set
`sheet_layout` to *separate frames* for the batch instead, which is what
`SaveImage` needs to write one file per arm. `levels` hands one arm on for
saving or comparing, *TF Save Report* keeps the table and *TF Save Images* keeps
the sheet — the sheet being the half that cannot be recomputed, since only the
arm named by `output_arm` leaves the node as a trajectory. Cost is two re-samples and one decode per arm, and `arm_limit`
(advanced) refuses to start rather than let a mistyped `0-1000` hold the GPU.

Sweeping *l\** is the one axis that cannot hold everything else fixed: a
selection snapped to one level's regions is not a whole region at another. The
node keeps the **token set** fixed, which is what "the same edit at every level"
has to mean, and says so in the report rather than refusing.

It sweeps a **feature** edit by default. Wire a *TF Region Map* into `regions`
and it sweeps a **shape** edit instead — `target_tokens` are handed to the region
named by `source_tokens`, taking that whole region's mean. The map describes one
level, so `seed` and `strength` are available and `level (l*)` is refused with
that reason; everything else about the loop, including the per-arm baseline and
the spread, is identical.

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

**Warnings at startup.** Three appear on every run and all three are benign —
none of them is worth acting on:

| warning | why it is harmless |
|---|---|
| `Tensorflow library not found, tensorflow.io.gfile operations will use native shim calls` | `flax` prefers TensorFlow's filesystem layer and falls back to plain Python IO without it. The only thing lost is `gs://` (Google Cloud Storage) paths, and every weight this loads is a local file. Installing TensorFlow to silence it would add ~600 MB for no function. |
| `You need pytorch with cu130 or higher to use optimized CUDA operations` | ComfyUI's own quantisation kernels, which nothing here uses. torch stays on cu128 to share a CUDA major with `jax[cuda12]`. |
| `The given NumPy array is not writable ... converting it to a tensor` | The RAE decoder wrapping a read-only latent array. It only reads it. |

They are deliberately not suppressed: silencing third-party warnings by default
is how a real one gets missed later.

**`Address already in use` from `run_comfyui_slurm.sh`.** It clears its own leftovers at
startup and steps to the next free port if something else holds it, so this
should not recur — if it does, `ss -ltnp | grep 8188` names the process.

**The Painter node says "Node 2.0 only".** That widget exists only in ComfyUI's
new node rendering. **Settings (gear, bottom left) → search "Node 2.0" →
enable**, then reload the page. *TF Level Canvas* reads the setting and says so
in its own body when it is off.

**The first run takes minutes and looks like nothing is happening.** Loading the
checkpoint plus an XLA compile plus building the RAE decoder — once per ComfyUI
start; later runs take about a second. *TF Load Pipeline* now drives a progress
bar across those three stages and reports how long it took, so a slow first run
is visibly slow rather than indistinguishable from a hang. On a compute node
with a cold JAX compilation cache it can take longer than you expect; give it
two minutes before concluding anything is wrong.

**Something misbehaved and you want to know why.** Every session writes two logs
to `outputs/comfyui_logs/` (the last 20 of each are kept):

| file | what |
|---|---|
| `comfyui-<timestamp>-<pid>.log` | the server's own log — every node that ran, every error |
| `bridge-<timestamp>-<pid>.log` | `run_comfyui_slurm.sh`'s port bridge, if you used it |

`run_comfyui_slurm.sh` prints both paths on exit. This matters because the terminal stream
is gone the moment the window is: without these, "I pressed Run and nothing
happened" leaves nothing at all to read afterwards.

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

**An edit appears to do something, but you are not sure it was the edit.**
Re-sampling from an edited canvas is still sampling, so one before/after pair
cannot separate the edit from the seed. *TF Sweep Edit* on the `seed` axis runs
the same edit across several seeds against a no-edit baseline for each, and
reports the spread. Workflow 05 is that graph, ready to run.

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
