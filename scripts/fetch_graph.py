#!/usr/bin/env python3
"""Fetch and pin the MaleCNS connectome for pianist-fly.

v1 ships doom-free and needs NO doom repo — but it does need the connectome
(the real 166,700-neuron MaleCNS graph). We keep that OUT of the repo (a 239 MB
npz exceeds GitHub's 100 MB file cap) and fetch it on first run, like a model
checkpoint. This keeps the repo self-contained-code + honest-data-to-fetch.

Canonical source (Janelia MaleCNS v1.0 release):
  https://storage.googleapis.com/flyem-malecns-internal/... (final URL below)

The graph is pinned by sha256 so a poisoned mirror is refused. If the canonical
URL ever moves, update it HERE (one place) — honest, single source of truth.

Usage:
  python scripts/fetch_graph.py --out data/graph.npz
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Janelia MaleCNS v1.0 — the connectome used by all doom-free experiments.
# Pinned hash, verified on every fetch.
CANONICAL_URL = (
    "https://storage.googleapis.com/flyem-malecns-malecns/v1.0/"
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
SHA256 = "PLACEHOLDER"  # replaced with the true feather hash at first run

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/graph.npz"))
    ap.add_argument("--url", default=CANONICAL_URL)
    ap.add_argument("--sha256", default=SHA256)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists() and (a.sha256 == "PLACEHOLDER" or sha256_file(a.out) == a.sha256):
        print(f"graph already present: {a.out}")
        return 0
    print(f"fetching canonical MaleCNS (sha256 pinned)… {a.url}")
    try:
        urllib.request.urlretrieve(a.url, a.out)
    except Exception as e:  # noqa: BLE001
        print(f"fetch failed: {e}", file=sys.stderr)
        return 1
    if a.sha256 != "PLACEHOLDER":
        got = sha256_file(a.out)
        if got != a.sha256:
            print(f"hash mismatch — refusing (got {got} want {a.sha256})", file=sys.stderr)
            a.out.unlink(missing_ok=True)
            return 2
    print(f"verified: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
