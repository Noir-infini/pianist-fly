#!/usr/bin/env python3
"""Rebuild the MaleCNS connectome `data/graph.npz` from the canonical Janelia
feathers — the honest, self-contained way to obtain the graph the LIF brain runs
on. The web never ships a `graph.npz`; that file is a COMPILED artifact.

Inputs (3 canonical MaleCNS v1.0 feathers, SHA-256 pinned):
  body-annotations-male-cns-v1.0-minconf-0.5.feather
  body-neurotransmitters-male-cns-v1.0.feather
  connectome-weights-male-cns-v1.0-minconf-0.5.feather

The compile is a faithful port of the reference pipeline that produced the
original graph (normalize -> index edges -> CSR compile + retinal projection),
so the output is byte-identical to the vendored graph when fed the same
feathers. Every ID is handled as an exact 64-bit integer; synapse counts are
positive uint32; weights are contact counts signed by the declared transmitter
and scaled by 0.275 (the reference's fixed per-contact factor, declared).

Usage:
  python scripts/build_graph_npz.py
      --annotations       data/mcns_annotations.feather
      --neurotransmitters data/mcns_neurotransmitters.feather
      --edges             data/mcns_edges.feather
      --out               data/graph.npz
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# Canonical MaleCNS v1.0 release files (Janelia flat-connectome tables). These
# hashes come from the release lockfile and are verified before any use.
PINNED = {
    "annotations": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/"
               "connectome-data/flat-connectome/"
               "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 14_483_314,
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    },
    "neurotransmitters": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/"
               "connectome-data/flat-connectome/"
               "body-neurotransmitters-male-cns-v1.0.feather",
        "bytes": 43_282_834,
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    },
    "edges": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/"
               "connectome-data/flat-connectome/"
               "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "bytes": 1_051_241_946,
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    },
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_ids(values) -> np.ndarray:
    """Never round 64-bit biological IDs through floating point or JavaScript."""
    items = np.asarray(values)
    if items.dtype.kind == "f":
        raise ValueError("Neuron IDs must be integers or decimal strings, never floats.")
    if items.dtype.kind in "iu":
        if np.any(items < 0):
            raise ValueError("Neuron IDs cannot be negative.")
        return items.astype(np.uint64)
    text = [str(value) for value in items]
    if any(not value.isascii() or not value.isdecimal() for value in text):
        raise ValueError("Neuron IDs must be nonnegative decimal integers.")
    return np.asarray(text, dtype=np.uint64)


def transmitter_signs(transmitters, ambiguous_sign=1):
    """Declared coarse fast-transmission assumption; never deletes unknown edges.

    ACh +; GABA, glutamate, histamine -. A co-transmitter combination with only
    one fast sign uses that sign; conflicting, missing and modulator-only cells
    use the explicit sensitivity parameter. This is NOT receptor physiology.
    """
    if ambiguous_sign not in (-1, 1):
        raise ValueError("Ambiguous edges must remain active with sign +1 or -1.")
    signs, uncertain = [], []
    for value in transmitters:
        tokens = set(str(value).lower().split(","))
        fast = ({1} if "acetylcholine" in tokens else set()) | (
            {-1} if tokens & {"gaba", "glutamate", "histamine"} else set())
        ambiguous = len(fast) != 1
        signs.append(ambiguous_sign if ambiguous else next(iter(fast)))
        uncertain.append(ambiguous)
    return np.asarray(signs, dtype=np.int8), np.asarray(uncertain, dtype=bool)


def normalize_nodes(frame, nt_frame):
    """Filter MaleCNS annotations to the retained neuron candidates.

    Keeps every entry with an assigned neuronal superclass, excluding explicit
    Glia status; no restriction to Traced status or typed cells. Neurotransmitter
    comes from the consensus prediction table where available.
    """
    import pandas as pd

    source = exact_ids(frame.bodyId)
    retain = frame.superclass.notna() & frame.superclass.astype(str).ne("")
    quality = frame.statusLabel.astype(object).fillna("unknown")
    nonneural = frame.status.eq("Glia")
    retain = retain & ~nonneural
    if nt_frame is not None:
        nt_frame = nt_frame.set_index("body")
        if not nt_frame.index.is_unique:
            raise ValueError("Duplicate male neurotransmitter IDs.")
        predicted = frame.bodyId.map(nt_frame.consensus_nt)
    else:
        raise ValueError("Neurotransmitters feather required for the sign pipeline.")
    catalog = pd.DataFrame({
        "source_id": source,
        "retained": np.asarray(retain, dtype=bool),
        "object_kind": np.where(nonneural, "non_neuronal",
                                np.where(retain, "neuron_candidate", "unresolved_object")),
        "quality": np.asarray(quality),
        "superclass": np.asarray(frame.superclass),
        "cell_type": np.asarray(frame.type),
        "neurotransmitter": np.asarray(predicted),
    }).sort_values("source_id", ignore_index=True)
    nodes = catalog.loc[catalog.retained].reset_index(drop=True)
    nodes.insert(0, "node_index", np.arange(len(nodes), dtype=np.uint32))
    return catalog, nodes


def index_edges(ids, pre, post, counts):
    """Return retained indices plus a mask accounting for every excluded row."""
    if not len(ids) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Node IDs must be nonempty, unique, and sorted.")
    pre, post = exact_ids(pre), exact_ids(post)
    counts = np.asarray(counts)
    if len(pre) != len(post) or len(pre) != len(counts):
        raise ValueError("Edge columns have different lengths.")
    if (not np.all(np.isfinite(counts)) or np.any(counts < 1)
            or np.any(counts != np.floor(counts)) or np.any(counts > 2**32 - 1)):
        raise ValueError("Synapse counts must be positive uint32-compatible integers.")
    i, j = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    keep = (i < len(ids)) & (j < len(ids))
    keep &= ids[np.minimum(i, len(ids) - 1)] == pre
    keep &= ids[np.minimum(j, len(ids) - 1)] == post
    return i[keep].astype(np.uint32), j[keep].astype(np.uint32), counts[keep].astype(np.uint32), keep


def build(annotations: Path, neurotransmitters: Path, edges: Path, out: Path) -> dict:
    """Compile data/graph.npz from the three canonical feathers."""
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    out = Path(out)
    manifest = {"source_hashes": {}}
    for key, pin in PINNED.items():
        path = {"annotations": annotations,
                "neurotransmitters": neurotransmitters,
                "edges": edges}[key]
        size = (Path(path).stat().st_size,)
        digest = file_digest(Path(path))
        manifest["source_hashes"][key] = {"url": pin["url"], "bytes": size,
                                          "sha256": digest}
        if digest != pin["sha256"]:
            raise ValueError(
                f"{key} hash mismatch: got {digest} want {pin['sha256']}\n"
                f"  refusing to compile from an unverified source file: {path}")
    print(f"verified {len(PINNED)} source feathers (sha256 pinned)")

    frame = feather.read_table(str(annotations)).to_pandas()
    nt_frame = feather.read_table(str(neurotransmitters)).to_pandas()
    catalog, nodes = normalize_nodes(frame, nt_frame)
    ids = exact_ids(nodes.source_id)
    print(f"retained neuron candidates: {len(nodes)} / {len(catalog)} annotation rows")

    edge_columns = ("body_pre", "body_post", "weight")
    reader = ipc.open_file(pa.memory_map(str(edges), "r"))
    stats = {key: 0 for key in
             ["source_edge_rows", "retained_edge_rows", "excluded_edge_rows",
              "source_synaptic_contacts", "retained_synaptic_contacts",
              "excluded_synaptic_contacts", "retained_weight_one_edges",
              "retained_self_edges"]}
    retained_pre, retained_post, retained_count = [], [], []
    incoming, outgoing = np.zeros(len(ids), dtype=np.int64), np.zeros(len(ids), dtype=np.int64)
    for number in range(reader.num_record_batches):
        batch = reader.get_batch(number)
        pre, post, weights = [batch.column(batch.schema.get_field_index(c))
                              .to_numpy(zero_copy_only=False) for c in edge_columns]
        i, j, count, keep = index_edges(ids, pre, post, weights)
        stats["source_edge_rows"] += len(pre)
        stats["retained_edge_rows"] += len(i)
        stats["source_synaptic_contacts"] += int(weights.sum(dtype=np.uint64))
        stats["retained_synaptic_contacts"] += int(count.sum(dtype=np.uint64))
        stats["retained_weight_one_edges"] += int(np.count_nonzero(count == 1))
        stats["retained_self_edges"] += int(np.count_nonzero(i == j))
        np.add.at(incoming, j, count)
        np.add.at(outgoing, i, count)
        retained_pre.append(i)
        retained_post.append(j)
        retained_count.append(count)
    pre = np.concatenate(retained_pre)
    post = np.concatenate(retained_post)
    count = np.concatenate(retained_count)
    stats["excluded_edge_rows"] = stats["source_edge_rows"] - stats["retained_edge_rows"]
    stats["excluded_synaptic_contacts"] = (stats["source_synaptic_contacts"]
                                           - stats["retained_synaptic_contacts"])
    assert int(incoming.sum()) == int(outgoing.sum()) == stats["retained_synaptic_contacts"]

    # CSR encodes every retained edge (self edges and weak edges included).
    order = np.argsort(pre, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=len(nodes)))].astype(np.int64)
    signs, uncertain = transmitter_signs(nodes.neurotransmitter)
    weight = (count[order].astype(np.float32) * signs[pre[order]] * 0.275).astype(np.float32)

    # Retinal projection: R1-R6 photoreceptors -> L1/L2/L3 lamina anchors.
    a = frame.set_index("bodyId").loc[nodes.source_id]
    receptor = a.type.eq("R1-R6").to_numpy()
    anchors = a.type.isin(["L1", "L2", "L3"]).to_numpy() & a.assignedOlHex1.notna().to_numpy()
    selected = receptor[pre] & anchors[post]
    cols = {}
    for i, j, w in zip(pre[selected], post[selected], count[selected]):
        key = (float(a.assignedOlHex1.iloc[j]), float(a.assignedOlHex2.iloc[j]))
        cols.setdefault(int(i), {})
        cols[int(i)][key] = cols[int(i)].get(key, 0) + int(w)
    indices, xy, confidence, hexes = [], [], [], []
    for i, counts in sorted(cols.items()):
        h = max(counts, key=counts.get)
        total = sum(counts.values())
        indices.append(i)
        hexes.append(h)
        confidence.append(counts[h] / total)
        xy.append((h[0] - 0.5 * h[1], np.sqrt(3) / 2 * h[1]))
    xy = np.asarray(xy)
    indices = np.asarray(indices, dtype=np.int32)
    uv = np.empty_like(xy)
    sides = a.rootSide.to_numpy()[indices]
    for side in ("L", "R"):
        mask = sides == side
        z = xy[mask]
        z = (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0))
        uv[mask, 0] = (0.60 * z[:, 0] if side == "L" else 0.40 + 0.60 * (1 - z[:, 0]))
        uv[mask, 1] = 1 - z[:, 1]

    readouts = []
    motor_types = ["DNa02", "DNp09", "MDN", "MN9", "DNp20", "DNpe017"]
    for i in np.flatnonzero(nodes.cell_type.isin(motor_types).to_numpy()):
        readouts.append({"index": int(i), "id": str(int(nodes.source_id.iloc[i])),
                         "type": str(nodes.cell_type.iloc[i]),
                         "side": str(a.somaSide.iloc[i])})

    manifest.update({
        "dataset": "malecns_v1",
        "sex": "male",
        "release": "MaleCNS v1.0",
        "coverage": "brain_and_ventral_nerve_cord",
        "source_annotation_rows": int(len(catalog)),
        "retained_neuron_candidates": int(len(nodes)),
        "graph": stats,
        "retina_total": int(np.count_nonzero(receptor)),
        "retina_mapped": int(len(indices)),
        "retina_unmapped": int(np.count_nonzero(receptor) - len(indices)),
        "projection_confidence_median": float(np.median(confidence)),
        "projection_below_80_percent": int(np.sum(np.asarray(confidence) < 0.8)),
        "readouts": readouts,
        "uncertain_sign_neurons": int(uncertain.sum()),
        "node_policy": "Every entry with an assigned superclass, including "
                       "uncertain tbc classes; exclude explicit Glia status; no "
                       "restriction to Traced status or typed cells.",
        "edge_policy": "All released edges between retained entries; no "
                       "additional weight threshold or removal of self-connections.",
        "upstream_filters": [
            "Published pre/post synapse confidence threshold 0.5",
            "Only annotated neuronal entries become simulation nodes; other "
            "segmentation objects remain in raw data",
        ],
        "upstream_autapses_excluded": False,
        "additional_edge_strength_threshold": None,
        "synaptic_weights_are_contact_counts": True,
        "retina_model": "R1-R6 luminance-only. Column inferred from all contacts "
                        "onto annotated L1/L2/L3; modal column. Experimental "
                        "overlapping viewport projection, not calibrated retinal "
                        "angles.",
        "visual_dynamics": "Photoreceptors and lamina are graded in vivo. This "
                           "experiment uses an explicit LIF proxy, low-pass "
                           "luminance drive and tonic lamina current; it is not "
                           "validated fly vision.",
        "training": "No synaptic plasticity. Optional score-contingent sugar "
                    "input is stimulation, not learning.",
    })

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             ptr=ptr, post=post[order].astype(np.int32), weight=weight,
             ids=nodes.source_id.to_numpy(dtype=np.int64),
             retina=indices, uv=uv.astype(np.float32), confidence=confidence,
             hexes=np.asarray(hexes),
             lamina=np.flatnonzero(nodes.cell_type.isin(["L1", "L2", "L3", "L5"])
                                   .to_numpy()).astype(np.int32),
             sugar=np.flatnonzero(nodes.cell_type.eq("LB3c").to_numpy()).astype(np.int32),
             superclass=np.asarray(nodes.superclass.fillna("unassigned").astype(str),
                                   dtype="U64"))
    print(f"wrote {out}  ({len(nodes)} neurons, {len(post[order])} edges, "
          f"{out.stat().st_size / 1e6:.1f} MB)")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, default=Path("data/mcns_annotations.feather"))
    ap.add_argument("--neurotransmitters", type=Path, default=Path("data/mcns_neurotransmitters.feather"))
    ap.add_argument("--edges", type=Path, default=Path("data/mcns_edges.feather"))
    ap.add_argument("--out", type=Path, default=Path("data/graph.npz"))
    ap.add_argument("--manifest", type=Path, default=Path("data/graph-manifest.json"))
    a = ap.parse_args()
    for label, path in (("annotations", a.annotations), ("neurotransmitters", a.neurotransmitters),
                        ("edges", a.edges)):
        if not Path(path).exists():
            ap.error(f"{label} feather not found: {path}")
    manifest = build(a.annotations, a.neurotransmitters, a.edges, a.out)
    if a.manifest:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {a.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())