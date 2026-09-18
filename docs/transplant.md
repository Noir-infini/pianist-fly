# Scent-drive transplant: doom removed, channel rebuilt, honesty locked

Date: 2026-09-18. This doc answers "did scent stay real?" with the code.

## 1. The only two files that ever touched doom — and what we did

The original `~/fly` project's press/music loop crossed a repo boundary twice.
Both were **decoder/conductor** code (the "conductor" handles light as an
instruction; a decoder aims the puppet), and both are now gone:

| file | what it did | status here |
| --- | --- | --- |
| `src/train_leg_readout.py:30` | trained a leg readout on doom.native | **doom removed** → now uses our own `native_brain.py` (NativeBrain contract) |
| `src/malecns_driver.py:42-43` | doom.engine NeuralControls mode="bci" (sight→decode) + NativeBrain load | **doom removed** → `native_brain.py` + `data/graph.npz` fetched by setup.sh |

Everything else in the press path — `src/native_press.py`, `src/leg_ik.py`,
`src/mirror.py`, the 4-leg stand — was already **our own doom-free code**.
`rg "doom\." src/` now returns **zero** matches.

## 2. Scent channel: real, not invented

`native_brain.py` loads `mcns_annotations.feather` + `mcns_somas.npz` from our
repo and marks, by **superclass in our own feather**, the real olfactory cells:

- **scent** — `2,635` olfactory receptor neurons (ORNs) via superclass prefix
  `cb_sensory`/`ol_sensory` + `mcns` receptor match
- **antenna** — `932` more (`antenna`/`Johnston's`), Johnston's organ cells are
  the fly's *gift from the antenna*; sugar scent enters through the antenna.

**No trained readout, no IK aiming, no decoder-as-conductor.** The fly's own
connectome + its native 4-leg coupling press the key (§17/§18 kept native);
the press sensor physically verifies; only a verified press becomes a sound.

## 3. What stays honest (contract)

- `source.kind`: `predicted` (LIF prediction over the MaleCNS connectome —
  never claimed as measured firing)
- CANNON: **no doom import, no doom output, no doom engine**; only our own
  `data/graph.npz` (fetched, hash-pinned) + our own annotations
- drive: sugar enters via the **antenna** (real scent), not the eye (we have
  the olfactory cells; the "sugar through the eye" shortcut was rejected)
- press: **native** — the fly's own leg coupling presses; no IK, no decoder;
  the press sensor verifies contact of the fly's own foot
- sound: only on a verified press; cadence = the fly's own (never faked)

Provenance of the graph itself: `data/graph.npz` is the MaleCNS v1.0 male
connectome (Janelia release), fetched by `./setup.sh` with a pinned sha256;
`src/build_v1_subgraph.py` then selects the scent→sugar→leg pathway **by
superclass in our annotations** (the honest nose + native press), keeping real
ids and wiring, so the clone runs offline but never invents edges.
