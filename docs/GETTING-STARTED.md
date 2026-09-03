# Getting started, from scratch

For someone who has never opened ComfyUI. If you already use it, the
[README](../README.md) is the faster route.

By the end of this you will have generated an image, watched it being built in
four passes, and changed part of one of those passes to alter the result.

---

## What this is

**Trajectory Forcing** is an image generator that does not paint a picture in
one go. It builds one in four passes, coarse to fine:

| pass | called | what it decides |
|---|---|---|
| level 0 | object/background | roughly where the thing is |
| level 1 | parts | its major parts |
| level 2 | subparts | the parts of those |
| level 3 | fine | the final detail |

The unusual bit is that **every pass can be turned into a picture**, not just
the last. So you can watch the image being decided:

![four levels of one trajectory](img/trajectory.png)

And because you can see the intermediate steps, you can *change* one and let the
model carry on from there. Change something at level 2 and the model re-does
level 3 to fit — it does not paste a rectangle over the result:

![before and after an edit](img/edit.png)

**ComfyUI** is the program you do this in. It shows a program as a diagram —
boxes ("nodes") wired together. You do not write any code; you load a diagram,
change a couple of numbers, and press Run.

**This extension** adds the Trajectory Forcing boxes to ComfyUI, plus four
ready-made diagrams to start from.

---

## Before you start

You need an **NVIDIA GPU with at least 8 GB of memory**, on Linux or Windows.
There is no Mac or CPU option — the model is far too slow without a GPU to be
worth trying.

To check, open a terminal and run:

```bash
nvidia-smi
```

You want a table with a memory figure of 8000 MiB or more. If the command is not
found, you do not have NVIDIA drivers installed, and nothing below will work
until you do.

You also need about **25 GB of disk**: 11 GB for the software, and the rest for
model weights that download on first use.

> Working on a university cluster with Slurm instead of your own machine? Use
> the [Slurm route in the README](../README.md#option-b-a-slurm-cluster) — it is
> a different set of commands, but everything after "Your first run" is the same.

---

## Install

Four commands. The third one takes about ten minutes and prints a lot.

```bash
# 1. Get ComfyUI itself
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI

# 2. Get this extension and the Trajectory Forcing code it runs
git clone https://github.com/mervekocabas/TrajectoryForcing ~/TrajectoryForcing
git clone <this-repo> ~/ComfyUI-TrajectoryForcing

# 3. Tell ComfyUI where the extension is
ln -s ~/ComfyUI-TrajectoryForcing ~/ComfyUI/custom_nodes/ComfyUI-TrajectoryForcing

# 4. Build the Python environment (~10 min, ~11 GB)
cd ~/ComfyUI-TrajectoryForcing
TF_REPO=~/TrajectoryForcing bash env/setup.sh
```

Then start it:

```bash
TF_REPO=~/TrajectoryForcing ./run_comfyui.sh
```

Wait for this line — it can take half a minute, and the port is not open before
it appears:

```
To see the GUI go to: http://0.0.0.0:8188
```

Open **http://localhost:8188** in a browser. You should see an empty grey
canvas. That is ComfyUI.

<details>
<summary>If something went wrong</summary>

**`ERROR: CUDA is not available`** — the GPU is not visible to Python. Check
`nvidia-smi` works, and that you ran `env/setup.sh` on the same machine.

**`Could not find the TrajectoryForcing checkout`** — set `TF_REPO` to wherever
you cloned it, as in the commands above. Put it in your shell profile if you
get tired of typing it.

**The page does not load** — make sure you waited for the "To see the GUI go
to" line. If the terminal is still printing, it is not ready.

</details>

---

## Your first run

**1. Load a diagram.** In ComfyUI, the **Workflow** menu (top left) →
**Open**, then pick:

```
~/ComfyUI-TrajectoryForcing/workflows/01-generate-and-decode.json
```

You will see a row of boxes and a yellow note on the left explaining them.

**2. Press Run.** The button is at the bottom of the screen. Nothing appears to
happen for **one to two minutes** — this is the model loading, and it only
happens once. Later runs take about a second.

**3. Look at the result.** Two strips of four images appear. The top one is the
image at each of the four passes, decoded to pixels. The bottom is the same four
passes shown as raw data (false colour) — that view matters later, because it is
what the editing works on.

![the latent view](img/latents.png)

**4. Change something.** Find the **TF ImageNet Class** box and pick a different
animal or object from its dropdown. Press Run again. This time it takes about a
second.

You have generated an image. Now the interesting part.

---

## Your first edit

Open `workflows/02-feature-edit-coords.json` the same way and press Run.

It generates **two** images — a *target* being edited and a *source* the new
content comes from — takes a patch of the source, writes it into a region of the
target at level 2, and re-runs level 3 from there. You get the edited result and
the unedited original side by side.

**The one thing to understand:** the edit happens on a **16×16 grid**. Every
image is described by 256 "tokens" at each level, and a token is the smallest
thing you can change. The coordinates in the *target region* box are `row,col`
on that grid — `7,7` is the middle.

Change `7,7` to something else and press Run. The preview labelled *what was
selected* shows exactly which tokens your coordinates picked, so if a result
surprises you, look there first.

Then try:

- **TF Feature Edit → level.** `2` is the default. Try `1` for a bigger, more
  semantic change, `3` for a small local one. Lower levels change more, because
  more passes get re-done afterwards.
- **strength.** `1.0` replaces outright; `0.5` blends.
- **the two class ids** in the target and source boxes.

When typing coordinates gets tedious, `workflows/03-feature-edit-painter.json`
lets you paint the region with a brush instead. Read its yellow note first — it
needs two runs, and that is deliberate.

---

## Where to go next

- **[workflows/README.md](../workflows/README.md)** — what each of the four
  diagrams does and what to change in it.
- **[README.md](../README.md)** — every node, what its settings mean, and the
  Slurm route.
- **[The paper](https://arxiv.org/abs/2606.22527)** — the method itself.

---

## Words you will see

| word | meaning |
|---|---|
| **node** | one box in the diagram |
| **workflow** | the whole diagram; a `.json` file |
| **widget** | a setting inside a box (a number, a dropdown) |
| **level** | one of the four passes, 0 (coarse) to 3 (fine) |
| **token** | one cell of the 16×16 grid — the smallest editable unit |
| **latent** | the model's internal representation, before it becomes pixels |
| **trajectory** | all four levels of one image, together |
| **region** | a clump of neighbouring tokens the model treats as one thing |
| **queue** | ComfyUI's list of runs waiting to happen |
