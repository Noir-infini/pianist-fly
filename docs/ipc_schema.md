# File IPC & JSON Schema Specification

`pianist-fly` components run as separate processes that communicate exclusively via atomic JSON file swaps. This document details the schema for `FLY_ACTIVITY` and `FLY_CMD`.

---

## 1. `FLY_ACTIVITY` (`malecns_activity.json`)

Written by `src/brain_driver.py` every $28.6\text{ ms}$ sim tick. Read by `play_piano.py` and `gnat-brain-male`.


### JSON Schema & Example
```json
{
  "t_sim": 1.43,
  "senses": {
    "scent": {
      "cL": 0.382,
      "cR": 0.124
    }
  },
  "intent": {
    "known": true,
    "bearing_deg": -28.4,
    "strength": 0.85,
    "firing": true
  },
  "piano_decision": {
    "decision": "none",
    "lateral": 0.42,
    "motor": {
      "T1L": 18.4,
      "T1R": 52.1
    }
  },
  "active": [
    {"id": 1042, "n": 4, "hz": 140.0},
    {"id": 8921, "n": 2, "hz": 70.0}
  ],
  "piano": {
    "note": "E4",
    "song_i": 2,
    "song_n": 34,
    "pressed_ok": true,
    "force_scale": 0.95,
    "t": 12.4
  }
}
```

### Key Fields
- **`senses.scent`**: Antennal concentration inputs ($c_L, c_R$).
- **`intent`**: Connectome orientation state ($bearing$, $strength$).
- **`piano_decision.motor`**: Motor pool firing rates ($T1L, T1R$ in Hz) used to scale leg press force.
- **`piano_decision.decision`**: Always `"none"` in the live driver — the brain never names a key; the note comes from `play_piano.py`'s own score, and the brain only shapes side/force via `motor`.
- **`active`**: Top active neurons with per-cell firing rates ($Hz$) for the 3D Rust visualizer.
- **`piano`**: Merged back by `play_piano.py` to publish real-time keypress verification status.

---

## 2. `FLY_CMD` (`pianist_cmd.json`)

Written by `src/play_piano.py` to publish target sugar plume locations to the brain driver.


### JSON Schema & Example
```json
{
  "scent": true,
  "scent_strength": 0.8,
  "source_x": 0.22,
  "source_y": -0.0979,
  "hunger": 0.75
}
```

### Key Fields
- **`scent`**: Boolean flag toggling plume presence.
- **`scent_strength`**: Plume magnitude coefficient.
- **`source_x` / `source_y`**: World coordinates of the target sugar plume (keybed offset $y$ with sign flip).
- **`hunger`**: Optional motivation override ($0.0 \dots 1.0$).
