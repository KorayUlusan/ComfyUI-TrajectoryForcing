# The example workflows

Four diagrams, in the order they are meant to be read. Each one is also
self-documenting: open it and there is a yellow note on the left saying what you
are looking at, and the nodes are boxed into numbered groups.

Load one with **Workflow → Open** in ComfyUI, or drag the `.json` onto the
canvas.

> New to ComfyUI? Read **[../docs/GETTING-STARTED.md](../docs/GETTING-STARTED.md)**
> first — it covers installing and pressing Run.

| | what it shows | runs as-is? |
|---|---|---|
| [01 · generate and decode](#01--generate-and-decode) | the method, no editing | yes |
| [02 · feature edit, typed](#02--feature-edit-coords) | change a region's content | yes |
| [03 · feature edit, painted](#03--feature-edit-painter) | the same, with a brush | needs two runs |
| [04 · shape edit](#04--shape-edit) | move a region's boundary | yes |

Everything here is generated from the live node definitions by
`scripts/make_workflows.py`, so the files cannot drift from what the nodes
actually accept. Each is written twice: the `.json` here for the ComfyUI menu,
and `api/<name>.json` for `POST /prompt`.

---

## 01 · generate and decode

Sample one trajectory and look at all four levels — decoded to pixels, and as
the raw token grid.

![four levels](../docs/img/trajectory.png)

The first run takes **1–2 minutes** (loading the model and compiling); every
later run takes about a second.

**Groups:** load the model → sample a trajectory → look at every level.

**Worth changing:**

| node → setting | effect |
|---|---|
| *TF ImageNet Class* | which of the 1000 ImageNet classes to generate |
| *TF Generate* → `seed` | a different sample of the same class |
| *TF Latent Preview* → `which` | one level instead of all four |

The lower strip is the PCA view — the same four levels as raw tokens in false
colour. It looks abstract, but it is what the edits in 02–04 operate on, and
region boundaries are far easier to see there than in the decoded image.

![the latent view](../docs/img/latents.png)

---

## 02 · feature edit (coords)

Take the average feature of a patch of one image, write it into a region of
another, and re-sample the finer levels from there.

![before and after](../docs/img/edit.png)

This is the paper's *feature edit*: `z̃ᵢ = f_src` for the selected tokens.
Because similar content sits near itself in the model's feature space, writing
one region's average onto another transfers its identity — and because the finer
levels are then re-generated rather than pasted, the result is coherent.

**Groups:** load and generate two images → choose what to edit → apply and
re-sample → compare → measure.

The edit's `level` is wired from *TF Region Map* rather than typed a second
time. A selection carries the level whose regions it was snapped to, and the
edit nodes refuse it at any other — the token grid is the same size at every
level, so otherwise the edit lands on the wrong region silently.

**The coordinates.** Selections are `row,col` on the **16×16 token grid**.
`7,7` is the middle. `6,6:9` means row 6, columns 6 through 9. The *target
region* input has the region map wired into it, so a single coordinate expands
to the whole region containing it; the *source tokens* input does not, so it
takes your coordinates literally.

**Worth changing:**

| node → setting | effect |
|---|---|
| *target region* → `coords` | **where** the edit lands |
| *source tokens* → `coords` | **what** gets written there |
| *TF Feature Edit* → `level` | 0–3. Lower = broader, more semantic change |
| *TF Feature Edit* → `strength` | below 1.0 blends instead of replacing |
| *TF Feature Edit* → `source_mode` | `region mean` is the paper's edit; `token cycle` copies token-for-token |
| either *TF Generate* → `class_id` | which two images you are mixing |

**If a result surprises you,** look at the *what was selected* preview before
changing anything else. It draws your selection on the level you edited, and
most surprises are the selection not being where you thought. Every preview is
numbered along its top and left edges, so a coordinate can be read off rather
than counted.

**Did it actually change anything?** *TF Compare Levels* at the bottom reports
tokens changed per level and draws a heatmap of where. An edit at level 2 should
show zero change at levels 0 and 1, the selected tokens at level 2, and diffuse
change at level 3. Anything else is the interesting kind of wrong.

**Levels below the edit never change.** Sampling is Markov in the level index,
so an edit only ever propagates upward. Editing at level 3 changes nothing
downstream of itself, because there is nothing above it — which is why the
default is 2.

---

## 03 · feature edit (painter)

The same edit, but you pick the region with a brush.

**This one takes two runs, by design.** ComfyUI has to run the graph once to
produce the canvas before there is anything to paint on:

1. **Run.** It stops at *TF Tokens From Mask* with a note saying nothing is
   painted yet. The canvas appears **inside the Painter node**.
2. **Paint** over the area you want to change.
3. **Run again.** Now it finishes.

> **"Node 2.0 only" in the Painter?** That widget exists only in ComfyUI's new
> node rendering. **Settings (gear, bottom left) → search "Node 2.0" → enable**,
> then reload. *TF Level Canvas* detects the setting and says so too. Workflow
> 02 needs none of this.

The Painter shows the canvas behind your brush because *TF Level Canvas*
publishes it as a node preview — the Painter takes its backdrop from the stored
preview of whatever feeds its `image` slot, not from the wire.

The grid on the canvas is the token grid — one cell is one token. The yellow
lines are region boundaries. Your stroke is **snapped to whole regions**, so it
does not need to be neat: covering most of a region selects all of it.

![the regions](../docs/img/regions.png)

**If it still stops after you painted,** the note in *TF Tokens From Mask* says
which of the two thresholds you missed:

| setting | meaning | when to change it |
|---|---|---|
| `coverage` | fraction of a token's pixels that must be painted | you painted too thinly |
| `region_overlap` | fraction of a region's tokens that must be painted | your stroke spread over too many regions |

Unwire the *regions* input from *TF Tokens From Mask* to select painted tokens
literally instead of snapping — useful for a deliberately ragged selection.

---

## 04 · shape edit

Change *where a region ends* rather than what it contains.

Tokens are handed from one region to a neighbour and take on the receiving
region's average feature, so the boundary moves and no new content is invented.
This is the paper's *shape edit*: reassign `R_a → R_b`.

**Groups:** load and generate → find a boundary and name both sides → move it →
result.

Two selections, both on the token grid:

- **tokens to hand over** — the ones that change sides.
- **one token in the receiving region** — anywhere inside the region taking
  them. The whole region it names supplies the feature, not just that token,
  which is what keeps the content unchanged.

They must name **different** regions; the node stops and says so otherwise.

**Look at the region map preview first.** It is the only reliable way to pick
coordinates that actually straddle a boundary — at the default threshold of 0.9
there are around 50 regions over the 256 tokens, and they are not where you
would guess.

**Worth changing:** *TF Region Map* → `cosine_threshold`. Higher splits into
more, smaller regions; lower merges them. It changes what counts as "one thing"
and therefore what a shape edit can move.

---

## Building your own

Useful pieces not wired into any of the four:

| node | for |
|---|---|
| **TF Tokens Combine** | union/intersect/subtract several selections into one region |
| **TF Compare Levels** | tokens changed per level between two trajectories, plus a heatmap — the number a writeup wants |
| **TF Save / Load Levels** | keep a trajectory across restarts, so two edits can be compared against the identical starting image |
| **TF Levels Info** | the class, seed and full edit history of a trajectory |
| **TF Level Canvas** → `view: decoded RGB` | paint against the picture instead of the false-colour tokens |

Three things to know when wiring your own graph:

- **The `pipeline` socket is usually not needed.** A trajectory carries the
  pipeline that made it. Wire it only for a trajectory from *TF Load Levels*, or
  to override.

- **Edits do not sample.** *TF Feature Edit* and *TF Shape Edit* only produce
  the edited canvas. Nothing happens to the image until *TF Resume From Level*
  re-generates the finer levels. A trajectory that has been edited but not
  resumed is marked, and *TF Decode Levels* warns rather than silently showing
  you stale levels.
- **Leave the advanced `-1`s alone unless you mean it.** `-1` means "decide for
  me" everywhere it appears: *TF Resume From Level* takes the level from
  whichever edit fed it and keeps the trajectory's class, *TF Feature Edit*
  reads the source from the level being edited. Each node prints which it did.
