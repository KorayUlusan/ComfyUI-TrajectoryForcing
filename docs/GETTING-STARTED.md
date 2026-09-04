# Getting started, from scratch

For someone who has never opened ComfyUI. If you already use it, the
[README](../README.md) is faster.

By the end you will have generated an image, watched it being built in four
passes, and changed one of those passes to alter the result.

---

## What this is

**Trajectory Forcing** does not paint a picture in one go. It builds one in four
passes, coarse to fine:

| pass | decides |
|---|---|
| level 0 | roughly where the thing is |
| level 1 | its major parts |
| level 2 | the parts of those |
| level 3 | the final detail |

The unusual bit: **every pass can be turned into a picture**, not just the last.
So you can watch the image being decided.

![four levels of one trajectory](img/trajectory.png)

And because you can see the intermediate steps, you can change one and let the
model carry on from there. Change something at level 2 and it re-does level 3 to
fit — it does not paste a rectangle over the result.

![before and after an edit](img/edit.png)

**ComfyUI** is the program you do this in: a diagram of boxes ("nodes") wired
together. You write no code — load a diagram, change a couple of numbers, press
Run. **This extension** adds the Trajectory Forcing boxes, plus five ready-made
diagrams.

---

## Before you start

An **NVIDIA GPU with at least 8 GB**, on **Linux** (Windows only inside WSL2 —
the JAX build this needs has no native-Windows version). No Mac or CPU option.

```bash
nvidia-smi     # want a memory figure of 8000 MiB or more
```

If that command is not found you have no NVIDIA drivers, and nothing below will
work until you do. You also need **~25 GB of disk**: 11 GB software, the rest
model weights that download on first use.

> On a university cluster with Slurm? Use the Slurm route in
> [README → Install](../README.md#install) — different commands, but everything
> from "Your first run" on is the same.

---

## Install

```bash
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
git clone https://github.com/KorayUlusan/ComfyUI-TrajectoryForcing ~/ComfyUI-TrajectoryForcing
ln -s ~/ComfyUI-TrajectoryForcing ~/ComfyUI/custom_nodes/ComfyUI-TrajectoryForcing

cd ~/ComfyUI-TrajectoryForcing
bash env/setup.sh      # ~10 min, ~11 GB, prints a lot
./run_comfyui.sh
```

Then open **http://localhost:8188**. An empty grey canvas is ComfyUI.

The Trajectory Forcing model code is not in that list — the extension fetches it
itself, pinned to a known-good version. Already have a copy? `cp .env.example .env`
and set `TF_REPO` in it.

<details>
<summary>If something went wrong</summary>

**`ERROR: CUDA is not available`** — the GPU is not visible to Python. Check
`nvidia-smi` works and that you ran `env/setup.sh` on this machine.

**`Could not find the TrajectoryForcing checkout`** — set `TF_REPO` to where you
cloned it.

Anything else: [README → Troubleshooting](../README.md#troubleshooting).
</details>

---

## Your first run

**1. Load a diagram.** **Workflow → Open** (top left), then pick:

```
~/ComfyUI-TrajectoryForcing/workflows/01-generate-and-decode.json
```

**2. Press Run** (bottom of the screen). The first run loads the model and takes
one to two minutes — a progress bar in *TF Load Pipeline* tracks it. Later runs
take about a second.

**3. Look at the result.** Two strips of four images: the top is each pass
decoded to pixels, the bottom is the same four as raw data in false colour. That
lower view matters later — it is what the editing works on.

![the latent view](img/latents.png)

**4. Change something.** Pick a different class in **TF ImageNet Class** and Run
again.

---

## Your first edit

Open `workflows/02-feature-edit-coords.json` and press Run.

It generates **two** images — a *target* being edited and a *source* the new
content comes from — takes a patch of the source, writes it into a region of the
target at level 2, and re-runs level 3 from there.

**The one thing to understand:** the edit happens on a **16×16 grid**. Each
image is 256 "tokens" at each level, and a token is the smallest changeable unit.

The *target region* box has that grid on it. **Click a cell** to choose where the
edit lands, or drag across several. The `row,col` text underneath is the same
selection written out — `7,7` is the middle — and you can type there instead.
Keeping it as text is what lets you write down exactly what you did.

The *what was selected* preview shows which tokens your coordinates actually
picked. Look there first if a result surprises you.

Then try:

- **TF Feature Edit → level.** `1` for a bigger, more semantic change, `3` for a
  small local one. Lower levels change more, because more passes get re-done after.
- **strength.** `1.0` replaces outright, `0.5` blends.
- **the two class ids**, to mix different things.

For a free-hand region instead of grid cells,
`workflows/03-feature-edit-painter.json` gives you a brush. It needs two runs,
and that is deliberate.

---

## "Was that the edit, or was that luck?"

A fair question, and the reason workflow 05 exists. Re-doing the finer passes
from an edited canvas is still *generating*, so it lands somewhere slightly
different every time — one before-and-after pair cannot tell you which part of
the difference you caused.

Open `workflows/05-sweep-seeds.json` and press Run. It makes the same edit with
four different seeds, each against an unedited run of that same seed, and reports
how far apart the four results are. Small spread means the edit decided the
picture; large means the seed did.

Details in [workflows/README.md](../workflows/README.md#05--sweep-seeds).

---

## Where to go next

- [**workflows/README.md**](../workflows/README.md) — what each diagram does and what to change.
- [**README.md**](../README.md) — every node and setting.
- [**The paper**](https://mervekocabas.github.io/TrajectoryForcing/) — the method itself.

---

## Words you will see

| word | meaning |
|---|---|
| **node** | one box in the diagram |
| **workflow** | the whole diagram; a `.json` file |
| **widget** | a setting inside a box |
| **level** | one of the four passes, 0 (coarse) to 3 (fine) |
| **token** | one cell of the 16×16 grid — the smallest editable unit |
| **latent** | the model's internal representation, before pixels |
| **trajectory** | all four levels of one image, together |
| **region** | a clump of neighbouring tokens the model treats as one thing |
