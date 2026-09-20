"""Build a gnat/desktop-fly style brain point cloud for the maleCNS.

Reads the cached MaleCNS annotations and writes the same JSON shape gnat and
desktop-fly use for their brain window:

    {"classes": ["ol_intrinsic", ...],
     "points": [[x, y, z, class_index, body_id], ...]}

Coordinates are centred on the brain and scaled so the cloud fits the renderer's
camera (roughly +/-7 units, same range as gnat's data/brain_points.json). The
trailing body_id lets the viewer look up where a spiking neuron is.

    .venv/bin/python scripts/build_brain_points.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATHER = ROOT / "data" / "mcns_annotations.feather"
OUT = ROOT / "data" / "mcns_brain_points.json"
RADIUS = 10.0  # target half-extent of the longest axis, in renderer units


def main():
    ann = pd.read_feather(FEATHER)
    rows = [(int(r.bodyId), r.somaLocation, r.superclass)
            for r in ann.itertuples()
            if r.somaLocation is not None and len(r.somaLocation) == 3]
    body = np.asarray([r[0] for r in rows], dtype=np.int64)
    pos = np.asarray([r[1] for r in rows], dtype=np.float64)
    sup = [str(r[2]) if r[2] is not None else "other" for r in rows]

    # Order classes by frequency so the palette's first colours are the big ones.
    counts = Counter(sup)
    classes = [name for name, _ in counts.most_common()]
    index = {name: i for i, name in enumerate(classes)}

    # Centre on the robust bounding box and scale the longest axis to +/-8, so
    # the cloud fills the renderer's camera the way gnat's data does. A radius
    # percentile would shrink the bulk of the brain to fit a few VNC outliers.
    lo = np.percentile(pos, 0.5, axis=0)
    hi = np.percentile(pos, 99.5, axis=0)
    centre = (lo + hi) / 2.0
    half = float(np.max((hi - lo) / 2.0))
    scale = RADIUS / half
    norm = ((pos - centre) * scale).astype(np.float32)

    points = [[round(float(norm[i, 0]), 3), round(float(norm[i, 1]), 3),
               round(float(norm[i, 2]), 3), index[sup[i]], int(body[i])]
              for i in range(len(rows))]
    OUT.write_text(json.dumps({"classes": classes, "points": points,
                               "source": "MaleCNS v1.0 body-annotations somaLocation"}))
    print(f"wrote {OUT}")
    print(f"  points: {len(points)}  classes: {len(classes)}  size: {OUT.stat().st_size/1e6:.1f} MB")
    print("  scale:", round(float(scale), 5), "centre:", [round(float(x), 1) for x in centre])
    print("  class counts:", counts.most_common(8))


if __name__ == "__main__":
    main()
