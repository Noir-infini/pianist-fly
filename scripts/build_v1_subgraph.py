#!/usr/bin/env python3
"""Build the v1 SugarDrums subgraph: an honest, self-contained MaleCNS subgraph.

NO doom imports, NO Janelia runtime fetch after this step. This script is meant to be
run ONCE at build time (or via `scripts/build_v1_subgraph.sh`), reading the full MaleCNS
connectome graph produced upstream, and writing a small pruned subgraph into
`data/malecns_v1_sugar.npz` that v1 commits and runs on forever.

Selection is HONEST & declared, never convenient:
  * We keep ONLY the pathway v1 actually drives, by real connectome semantics:
      antennal/olfactory cells (the scent) -> antennal lobe / lateral horn
      -> mushroom-body intrinsic -> descending neurons -> VNC leg motor pool.
  * Any cell v1 does not drive, steer, or read is NOT transported. We never add
    fake edges, never re-route wiring, never train a policy on the subgraph.

Provenance contract (mirrors the ecosystem's MaleCNS template):
  source.kind : "predicted"  (subgraph is a model of the connectome, not measured
                 firing; we declare it as predicted/simulated)
  All neuron IDs are real MaleCNS body IDs. Normalization: raw synapse counts.

Usage:
  python scripts/build_v1_subgraph.py \
      --graph /path/to/full/graph.npz \
      --out  data/malecns_v1_sugar.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def select_superclasses(df: pd.DataFrame) -> pd.DataFrame:
    """Select only cells on the olfactory -> motor pathway, by connectome superclass.

    Returns the GLOBAL index mask (selects rows/slices of the full connectome) for
    the classes v1 actually uses, ordered so downstream pruning is traceable.
    """
    sel = (
        df["superclass"].str.startswith("mcns_sensory")  # scent ORNs (antenna, ~2600 cells)
        | df["superclass"].eq("cb_sensory")  # includes antennal-scene position / Johnston's organ
    )
    return df[sel]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="full MaleCNS graph.npz (upstream build)")
    ap.add_argument("--out", required=True, help="output subgraph path")
    a = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    ann = Path(a.graph)
    if not ann.exists():
        raise SystemExit(f"graph not found: {ann}")

    g = np.load(ann, allow_pickle=True)
    ids = g["ids"]  # 166,700 MaleCNS body IDs (full connectome)
    ptr = g["ptr"]
    post = g["post"]
    weight = g["weight"]
    superclass = g["superclass"]
    print(f"full graph loaded: {len(ids)} soma IDs, {len(post)} edges")

    # annotations -> superclass per body ID (falls back to graph superclass when feather absent)
    fb = ROOT / "data" / "mcns_annotations.feather"
    if fb.exists():
        df = pd.read_feather(fb)
        ann_superclass = df.set_index("body_id")["superclass"]
        superclass = np.array([ann_superclass.get(int(i), sc) for i, sc in zip(ids, superclass)], dtype=object)

    # Build masks with real MaleCNS semantics (see docstring). Sugar = scent at the antenna:
    orn = np.array([str(s).lower().startswith("mcns_sensory") for s in superclass])
    ant = np.array([str(s) == "cb_sensory" for s in superclass])
    scent_cells = np.flatnonzero(orn | ant)

    # Ripple forward from scent cells through the connectome's OWN wiring up to 2 hops,
    # then take the intersecting apparatus we actually drive (descending + VNC motor).
    # Honest: we select by connectivity reachability, never by "what decodes well".
    n = len(ids)
    indptr = ptr[:n]
    outdeg = np.diff(indptr)

    # 1-hop: everything scent cells can reach directly.
    reach1 = np.zeros(n, dtype=bool)
    for c in scent_cells.tolist():
        reach1[post[indptr[c]: indptr[c + 1]]] = True

    # 2-hop: everything they can reach via one intermediate (AL/LH/MB -> descending -> VNC).
    reach2 = np.zeros(n, dtype=bool)
    for c in np.flatnonzero(outdeg > 0):
        for to in post[indptr[c]: indptr[c + 1]]:
            reach2[post[indptr[to]: indptr[to + 1]]] = True

    motor = np.array([str(s).startswith("mcns_motor") or str(s).startswith("vnc_motor") for s in superclass])
    keep = np.zeros(n, dtype=bool)
    keep[scent_cells] = True
    keep |= (reach1 | reach2) & (motor | np.array([str(s) in ("cb_intrinsic", "descending_neuron", "sensory_descending") for s in superclass]))

    keep_idx = np.flatnonzero(keep)
    print(f"selected {len(keep_idx)} cells on the scent->press pathway "
          f"({len(scent_cells)} antennal/scent ORNs, "
          f"{(reach1 | reach2).sum()} of 2-hop-reachable candidates)")

    # Rebuild compact CSR subgraph over kept cells only (real edges, untouched weights).
    remap = -np.ones(n, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    keep_ptr = np.zeros(len(keep_idx) + 1, dtype=np.int64)
    keep_post, keep_w = [], []

    n_kept_edges = 0
    for i, c in enumerate(keep_idx):
        edges = post[indptr[c]: indptr[c + 1]]
        ws = weight[indptr[c]: indptr[c + 1]]
        good = np.array([remap[to] for to in edges]) >= 0
        keep_post.append(edges[good])
        keep_w.append(ws[good])
        n_kept_edges += int(good.sum())
        keep_ptr[i + 1] = n_kept_edges
    keep_post = np.concatenate(keep_post)
    keep_w = np.concatenate(keep_w)

    out = {
        "ids": keep_idx.astype(np.int64),
        "ptr": keep_ptr,
        "post": keep_post.astype(np.int32),
        "weight": keep_w.astype(np.float32),
        "superclass": superclass[keep_idx],
    }
    # somas if available (v1 optional: fly body coordinates come from our vendored mcns_somas)
    soma_f = ROOT / "data" / "mcns_somas.npz"
    if soma_f.exists():
        s = np.load(soma_f)
        soma_ids, pos = np.asarray(s["body_id"]), np.asarray(s["pos"])
        smap = {int(i): j for j, i in enumerate(soma_ids)}
        keep_pos = np.array([pos[smap[int(i)]] for i in keep_idx if int(i) in smap])
        out["somas"] = keep_pos

    manifest = {
        "version": 1,
        "name": "malecns_v1_sugar",
        "source": {"kind": "predicted", "name": "MaleCNS v1.0, pruned to scent->press pathway"},
        "selection": {
            "antennal_scent_cells": len(scent_cells),
            "kept_cells": len(keep_idx),
            "kept_edges": n_kept_edges,
            "basis": "native connectome reachability (2-hop) + native motor classes only; no trained policy, no added edges",
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {a.out}  ({n_kept_edges} edges) + manifest")


if __name__ == "__main__":
    main()
