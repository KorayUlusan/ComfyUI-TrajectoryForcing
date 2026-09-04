# Getting started, from scratch

For someone who has never opened ComfyUI. If you already use it, the
[README](../README.md) is faster.

By the end you will have generated an image, watched it being built in four
passes, and changed one of those passes to alter the result.

---

## What this is

Trajectory Forcing does not paint a picture in one go. It builds one in four
passes, coarse to fine:

| pass | decides |
|---|---|
| level 0 | roughly where the thing is |
| level 1 | its major parts |
| level 2 | the parts of those |
| level 3 | the final detail |

The unusual part is that every pass can be turned into a picture, not just the
last. So you can watch the image being decided.

![four levels of one trajectory](img/trajectory.png)

And because you can see the intermediate steps, you can change one and let the
model carry on from there. Change something at level 2 and it re-does level 3 to
fit, rather than pasting a rectangle over the result.

![before and after an edit](img/edit.png)

ComfyUI is the program you do this in: a diagram of boxes, called nodes, wired
together. You write no code. You load a diagram, change a couple of numbers, and
press Run. This extension adds the Trajectory Forcing boxes, plus five ready-made
diagrams.

---

## Before you start

You need an NVIDIA GPU with at least 8 GB, on Linux. Windows works only inside
WSL2, because the JAX build this needs has no native-Windows version. There is no
Mac or CPU option.

```bash
nvidia-smi     # you want a memory figure of 8000 MiB or more
```

If that command is not found you have no NVIDIA drivers, and nothing below will
work until you do. You also need about 25 GB of disk: 11 GB for the software, and
the rest for model weights that download on first use.

> On a university cluster with Slurm? Use the Slurm route in
> [README → Install](../README.md#install). The commands differ, but everything
> from "Your first run" onwards is the same.

---

## Install

Everything goes in one folder. Pick where, and nothing is written outside it.
Uninstalling later means deleting that folder.

```bash
export TF_HOME="$HOME/trajectory-forcing"    # anywhere you like

python3.11 -m venv "$TF_HOME/cli"
"$TF_HOME/cli/bin/pip" install comfy-cli
"$TF_HOME/cli/bin/comfy" --workspace "$TF_HOME/ComfyUI" \
    install --nvidia --skip-torch-or-directml
"$TF_HOME/cli/bin/comfy" --workspace "$TF_HOME/ComfyUI" \
    node install --mode remote comfyui-trajectoryforcing

cd "$TF_HOME/ComfyUI/custom_nodes/comfyui-trajectoryforcing"
COMFY_DIR="$TF_HOME/ComfyUI" COMFY_VENV="$TF_HOME/venv" bash env/setup.sh   # ~10 min, ~11 GB
COMFY_DIR="$TF_HOME/ComfyUI" COMFY_VENV="$TF_HOME/venv" ./run_comfyui.sh
```

Then open **http://localhost:8188**. An empty grey canvas is ComfyUI.

Typing those two variables every time gets old: `cp .env.example .env`, put them
in there, and every script picks them up.

The Trajectory Forcing model code is not downloaded by any of the above. The
extension fetches it itself on first use, pinned to a known-good version. If you
already have a copy, set `TF_REPO` in that same `.env`.

<details>
<summary>If something went wrong</summary>

`ERROR: CUDA is not available` means the GPU is not visible to Python. Check that
`nvidia-smi` works and that you ran `env/setup.sh` on this machine.

`Could not find the TrajectoryForcing checkout` means you should set `TF_REPO` to
wherever you cloned it.

Anything else: [README → Troubleshooting](../README.md#troubleshooting).
</details>

---

## Your first run

**1. Load a diagram.** Use **Workflow → Open** at the top left, then pick:

```
$TF_HOME/ComfyUI/custom_nodes/comfyui-trajectoryforcing/example_workflows/01-generate-and-decode.json
```

**2. Press Run**, at the bottom of the screen. The first run loads the model and
takes one to two minutes, tracked by a progress bar in *TF Load Pipeline*. Later
runs take about a second.

**3. Look at the result.** Two strips of four images appear. The top one is each
pass decoded to pixels. The bottom is the same four as raw data in false colour,
and that lower view matters later, because it is what the editing works on.

![the latent view](img/latents.png)

**4. Change something.** Pick a different class in *TF ImageNet Class* and run
again.

---

## Your first edit

Open `example_workflows/02-feature-edit-coords.json` and press Run.

It generates two images, a target being edited and a source the new content comes
from. It takes a patch of the source, writes it into a region of the target at
level 2, and re-runs level 3 from there.

The one thing to understand is that the edit happens on a 16×16 grid. Each image
is 256 tokens at each level, and a token is the smallest thing you can change.

The *target region* box has that grid on it. Click a cell to choose where the edit
lands, or drag across several. The `row,col` text underneath is the same selection
written out, where `7,7` is the middle, and you can type there instead. Keeping it
as text is what lets you write down exactly what you did.

The *what was selected* preview shows which tokens your coordinates actually
picked. Look there first if a result surprises you.

Then try:

- *TF Feature Edit* → `level`. Use `1` for a bigger, more semantic change and `3`
  for a small local one. Lower levels change more, because more passes get re-done
  afterwards.
- `strength`. `1.0` replaces outright, `0.5` blends.
- the two class ids, to mix different things.

For a free-hand region instead of grid cells,
`example_workflows/03-feature-edit-painter.json` gives you a brush. It needs two runs, and
that is deliberate.

---

## "Was that the edit, or was that luck?"

A fair question, and the reason workflow 05 exists. Re-doing the finer passes from
an edited canvas is still generating, so it lands somewhere slightly different
every time. One before-and-after pair cannot tell you which part of the difference
you caused.

Open `example_workflows/05-sweep-seeds.json` and press Run. It makes the same edit with
four different seeds, each against an unedited run of that same seed, and reports
how far apart the four results are. A small spread means the edit decided the
picture. A large one means the seed did.

Details in [example_workflows/README.md](../example_workflows/README.md#05--sweep-seeds).

---

## Where to go next

- [**example_workflows/README.md**](../example_workflows/README.md) for what each diagram does and what to change.
- [**README.md**](../README.md) for every node and setting.
- [**The paper**](https://mervekocabas.github.io/TrajectoryForcing/) for the method itself.

---

## Words you will see

| word | meaning |
|---|---|
| **node** | one box in the diagram |
| **workflow** | the whole diagram, saved as a `.json` file |
| **widget** | a setting inside a box |
| **level** | one of the four passes, 0 (coarse) to 3 (fine) |
| **token** | one cell of the 16×16 grid, the smallest editable unit |
| **latent** | the model's internal representation, before pixels |
| **trajectory** | all four levels of one image, together |
| **region** | a clump of neighbouring tokens the model treats as one thing |
