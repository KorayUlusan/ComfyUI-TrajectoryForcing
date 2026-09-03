// A clickable 16x16 token grid for TF Tokens From Coords.
//
// Coordinates are typed as `row,col` on the token grid, and typing them means
// counting cells on a picture. Tick labels made a coordinate readable; this
// makes one enterable.
//
// Two things about the shape of this, both deliberate:
//
//   * It is a *widget on the existing node*, not a node of its own. The grid
//     writes into the node's own `coords` string, which stays the single source
//     of truth -- so a selection is still a text value you can paste into a
//     writeup, which is the entire reason TF Tokens From Coords exists.
//
//   * If this file fails to load, or ComfyUI changes an API it leans on, the
//     `coords` text field is still there and typing still works. That is the
//     whole bet: the extension gains a convenience, never a dependency. Nothing
//     here may ever become the only way to do something.
//
// When a TF Region Map is wired in, the node snaps whatever you picked to whole
// cosine regions (`min_overlap=0.0` -- a typed coordinate is a deliberate pick,
// not a rough stroke). The grid therefore has to select regions too, or it lies:
// you click one cell, it says "1 token", and the node selects the forty that
// share its region. The node hands its map back on `tf_regions` for exactly
// this. Until the graph has run once there is no map to hand back, so the grid
// starts token-wise and becomes region-wise after the first run -- the same
// one-round-trip shape the Painter workflow already has.
//
// It is a DOM widget rather than a Vue component on purpose -- core's own
// AUDIO_UI widget uses `addDOMWidget` unconditionally, so it renders under both
// the classic and the Node 2.0 renderers. Painter's "Node 2.0 only" problem does
// not apply here.
//
// `writeCoords` below must produce byte-identical output to
// `tf_nodes/tokens.py::format_coords`, which is the tested reference: the grid
// and the text field are two views of one value, and clicking must not silently
// rewrite what someone typed into something merely equivalent.

import { app } from "../../scripts/app.js";

const NODE = "TFTokensFromCoords";
const GRID = 16;        // every released checkpoint's token grid; see DEFAULT_GRID
const TICK_EVERY = 4;   // matches render.TICK_EVERY, so this reads like the previews

// Mirrors the _COORD regex in tokens.py: `r,c` with an optional `:c1` run.
const COORD = /(\d+)\s*,\s*(\d+)(?:\s*:\s*(\d+))?/g;

function readCoords(text, rows, cols) {
  const mask = Array.from({ length: rows }, () => new Array(cols).fill(false));
  for (const [, r, c0, c1] of (text || "").matchAll(COORD)) {
    const row = +r;
    if (row < 0 || row >= rows) continue;          // out of range: ignore here,
    const lo = Math.min(+c0, c1 === undefined ? +c0 : +c1);   // the node still
    const hi = Math.max(+c0, c1 === undefined ? +c0 : +c1);   // raises on it
    for (let col = Math.max(0, lo); col <= Math.min(cols - 1, hi); col++) {
      mask[row][col] = true;
    }
  }
  return mask;
}

function writeCoords(mask) {
  const parts = [];
  for (let row = 0; row < mask.length; row++) {
    for (let col = 0; col < mask[row].length; col++) {
      if (!mask[row][col]) continue;
      const start = col;
      while (col + 1 < mask[row].length && mask[row][col + 1]) col++;
      parts.push(col === start ? `${row},${start}` : `${row},${start}:${col}`);
    }
  }
  return parts.join(" ");
}

function buildGrid(node, coordsWidget) {
  const root = document.createElement("div");
  root.className = "tf-token-grid";
  Object.assign(root.style, {
    display: "flex", flexDirection: "column", gap: "4px",
    padding: "4px", boxSizing: "border-box", width: "100%",
  });

  const board = document.createElement("div");
  Object.assign(board.style, {
    display: "grid",
    gridTemplateColumns: `repeat(${GRID}, 1fr)`,
    gap: "1px",
    background: "#2a3550",
    border: "1px solid #3d4a6b",
    aspectRatio: "1 / 1",
    // Touch scrolling would otherwise steal the drag-to-paint gesture.
    touchAction: "none",
    userSelect: "none",
  });

  const status = document.createElement("div");
  Object.assign(status.style, {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    font: "11px monospace", color: "#94a3b8",
  });
  const count = document.createElement("span");
  const clear = document.createElement("button");
  clear.textContent = "clear";
  Object.assign(clear.style, {
    font: "11px monospace", color: "#94a3b8", background: "transparent",
    border: "1px solid #3d4a6b", borderRadius: "3px", cursor: "pointer",
    padding: "1px 6px",
  });
  status.append(count, clear);

  let mask = readCoords(coordsWidget.value, GRID, GRID);
  let regions = null;   // {ids: number[][], level, num_regions}, once the node has run
  const cells = [];

  // Every cell sharing a region id with (row, col). One click on a region-mapped
  // grid picks the whole thing, which is the paper's R_tgt -- a semantic part,
  // not an arbitrary set of tokens.
  const regionOf = (row, col) => {
    if (!regions?.ids?.[row]) return [[row, col]];
    const id = regions.ids[row][col];
    const out = [];
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) if (regions.ids[r]?.[c] === id) out.push([r, c]);
    }
    return out;
  };

  // A cell edge is a region edge when its neighbour has a different id. Drawn on
  // the grid so a region can be seen before it is clicked, matching the yellow
  // boundaries on TF Region Map's own preview.
  const edges = (row, col) => {
    if (!regions?.ids?.[row]) return "";
    const id = regions.ids[row][col];
    const differs = (r, c) => regions.ids[r]?.[c] !== undefined && regions.ids[r][c] !== id;
    const sides = [];
    if (row === 0 || differs(row - 1, col)) sides.push("inset 0 1px 0 0 #ffd664");
    if (col === 0 || differs(row, col - 1)) sides.push("inset 1px 0 0 0 #ffd664");
    if (row === GRID - 1 || differs(row + 1, col)) sides.push("inset 0 -1px 0 0 #ffd664");
    if (col === GRID - 1 || differs(row, col + 1)) sides.push("inset -1px 0 0 0 #ffd664");
    return sides.join(", ");
  };

  const paint = () => {
    let selected = 0;
    for (let row = 0; row < GRID; row++) {
      for (let col = 0; col < GRID; col++) {
        const on = mask[row][col];
        if (on) selected++;
        const cell = cells[row * GRID + col];
        cell.style.background = on ? "#ff5050" : "#151d2e";
        // Region boundaries first, because they are what a click follows; the
        // grid rule is only a ruler. Every fourth line, matching the tick labels
        // on the previews.
        const boundary = edges(row, col);
        const rule = row % TICK_EVERY === 0 || col % TICK_EVERY === 0;
        cell.style.boxShadow = boundary || (on ? "inset 0 0 0 1px #ffb0b0"
          : rule ? "inset 0 0 0 1px #33415c" : "none");
      }
    }
    const tokens = `${selected} token${selected === 1 ? "" : "s"}`;
    count.textContent = regions
      ? `${tokens} · ${regions.num_regions} regions at level ${regions.level}`
      : tokens;
  };

  const commit = () => {
    const text = writeCoords(mask);
    if (coordsWidget.value === text) return;
    coordsWidget.value = text;
    coordsWidget.callback?.(text);
    node.onWidgetChanged?.(coordsWidget.name, text, undefined, coordsWidget);
    app.graph?.setDirtyCanvas(true, false);
  };

  // Drag paints, and every cell it crosses is set to whatever the *first* cell
  // became -- otherwise dragging over a mixed area toggles cells back and forth
  // and the result depends on the exact path taken.
  let dragging = false;
  let dragTo = true;

  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      const cell = document.createElement("div");
      cell.title = `${row},${col}`;   // updated once a region map arrives
      cell.style.cursor = "pointer";
      cells.push(cell);
      board.append(cell);
    }
  }

  // Which cell is under the pointer, by hit-testing rather than by listening on
  // each cell. Per-cell `pointerenter` does not work here: the drag has to be
  // pointer-captured to survive leaving the cell it started in, and a capture
  // routes every later event to the capturing element, so siblings never see
  // `pointerenter` at all. Touch is worse -- the browser captures to the
  // pointerdown target implicitly, so it never worked there either. Capturing
  // on the *board* and asking the document what is under the cursor fixes both:
  // capture changes event routing, not hit-testing.
  const cellAt = (x, y) => cells.indexOf(document.elementFromPoint(x, y));

  // `wholeRegion` is off when a modifier is held (alt is option on a Mac), so a
  // single coordinate stays writable even on a region-mapped grid. Note what
  // this does and does not do: the node snaps with min_overlap=0.0, so the
  // *selection* is the whole region either way. What this controls is the
  // `coords` text -- "7,7" is a better line in a writeup than a nine-run
  // coordinate list, and it says which region was meant without ambiguity.
  // Unwire `regions` if you actually want sub-region tokens.
  let wholeRegion = true;

  const applyAt = (index) => {
    if (index < 0) return;
    const row = Math.floor(index / GRID);
    const col = index % GRID;
    const targets = wholeRegion ? regionOf(row, col) : [[row, col]];
    let changed = false;
    for (const [r, c] of targets) {
      if (mask[r][c] === dragTo) continue;
      mask[r][c] = dragTo;
      changed = true;
    }
    if (changed) paint();
  };

  board.addEventListener("pointerdown", (event) => {
    const index = cellAt(event.clientX, event.clientY);
    if (index < 0) return;
    event.preventDefault();
    event.stopPropagation();   // keep it off the canvas, or the node is dragged
    dragging = true;
    wholeRegion = !(event.altKey || event.ctrlKey || event.metaKey);
    dragTo = !mask[Math.floor(index / GRID)][index % GRID];
    applyAt(index);
    board.setPointerCapture?.(event.pointerId);
  });

  board.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    event.preventDefault();
    applyAt(cellAt(event.clientX, event.clientY));
  });

  const endDrag = (event) => {
    if (!dragging) return;
    dragging = false;
    if (event?.pointerId !== undefined) board.releasePointerCapture?.(event.pointerId);
    commit();
  };
  board.addEventListener("pointerup", endDrag);
  board.addEventListener("pointercancel", endDrag);
  // A pointer released outside the board would otherwise leave `dragging` true
  // and the next move would keep painting.
  window.addEventListener("pointerup", endDrag);

  clear.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    mask = readCoords("", GRID, GRID);
    paint();
    commit();
  });

  root.append(board, status);
  paint();

  // Typing into the text field is still first-class, so the grid follows it.
  return {
    root,
    resync: () => { mask = readCoords(coordsWidget.value, GRID, GRID); paint(); },
    // The node hands its region map back after each run. Shape-check it rather
    // than trusting it: a different checkpoint could use a different grid, and
    // a half-applied map would draw boundaries that are not there.
    setRegions: (payload) => {
      const ids = payload?.ids;
      const usable = Array.isArray(ids) && ids.length === GRID
        && ids.every((r) => Array.isArray(r) && r.length === GRID);
      regions = usable ? payload : null;
      for (let row = 0; row < GRID; row++) {
        for (let col = 0; col < GRID; col++) {
          cells[row * GRID + col].title = regions
            ? `${row},${col} — region ${regions.ids[row][col]} `
              + `(alt/option-click to write just this coordinate)`
            : `${row},${col}`;
        }
      }
      paint();
    },
  };
}

app.registerExtension({
  name: "TrajectoryForcing.TokenGrid",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onCreated?.apply(this, arguments);
      try {
        const coords = this.widgets?.find((w) => w.name === "coords");
        if (!coords) return result;      // schema changed; leave typing alone

        const { root, resync, setRegions } = buildGrid(this, coords);
        const widget = this.addDOMWidget("tf_token_grid", "div", root, {
          hideOnZoom: false,
          getMinHeight: () => 210,
        });
        // The coords string is the only thing worth saving; serialising the
        // grid too would put the same value in the workflow file twice, and a
        // stale copy is worse than none.
        widget.serialize = false;
        if (widget.options) widget.options.serialize = false;

        const previous = coords.callback;
        coords.callback = function (...args) {
          const out = previous?.apply(this, args);
          resync();
          return out;
        };
        // `onExecuted` is how a node receives its own `ui` payload back -- the
        // same hook core's AUDIO_UI widget uses. It is what carries the region
        // map, and it is a public, long-standing API rather than a store
        // internal, which matters for a file that has to keep working.
        const onExecuted = this.onExecuted;
        this.onExecuted = function (message) {
          const out = onExecuted?.apply(this, arguments);
          try {
            setRegions(message?.tf_regions?.[0] ?? null);
          } catch (error) {
            console.error("[TrajectoryForcing] region map unusable:", error);
          }
          return out;
        };
        // Loading a saved workflow sets widget values without firing callbacks.
        const onConfigure = this.onConfigure;
        this.onConfigure = function (...args) {
          const out = onConfigure?.apply(this, args);
          resync();
          return out;
        };
      } catch (error) {
        // Never take the node down with the convenience. Typing must survive
        // anything that goes wrong in here.
        console.error("[TrajectoryForcing] token grid unavailable:", error);
      }
      return result;
    };
  },
});
