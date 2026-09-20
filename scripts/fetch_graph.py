#!/usr/bin/env python3
"""Fetch and pin the MaleCNS connectome for pianist-fly.

The web never ships a `graph.npz` — Janelia publishes the MaleCNS v1.0 release
as flat-connectome `.feather` tables. `data/graph.npz` is a COMPILED artifact
(CSR re-indexed to the 166,700 retained neurons) built by
`scripts/build_graph_npz.py`.

This script:
  1. downloads the 3 canonical feathers (if not already present and verified),
  2. checks each against its pinned SHA-256 (a poisoned mirror is refused),
  3. invokes the compile to produce `data/graph.npz`.

The edges feather is ~1.05 GB (full-fidelity 25.6M retained edges); the
"significant-only" variants drop weak edges and would silently change the
brain, so we always use the full release. One-time download, then the compiled
`data/graph.npz` is used forever.

Usage:
  python scripts/fetch_graph.py
  python scripts/fetch_graph.py --out data/graph.npz
"""
import argparse
import hashlib
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CANONICAL = {
    "annotations": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/"
               "connectome-data/flat-connectome/"
               "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
        "local": "data/mcns_annotations.feather",
    },
    "neurotransmitters": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/"
               "connectome-data/flat-connectome/"
               "body-neurotransmitters-male-cns-v1.0.feather",
        "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
        "local": "data/mcns_neurotransmitters.feather",
    },
    "edges": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/"
               "connectome-data/flat-connectome/"
               "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
        "local": "data/mcns_edges.feather",
    },
}

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _progress(name: str, done: int, total: int, t0: float):
    """Live `\\r` download bar on stderr; degrades to plain MB when no size."""
    mb = done / 1e6
    speed = done / 1e6 / max(1e-6, time.monotonic() - t0)
    if total:
        pct = done / total * 100.0
        fill = 30 * done // total
        bar = "#" * fill + "-" * (30 - fill)
        line = (f"\r  {name:16s} {mb:7.1f} / {total/1e6:7.1f} MB "
                f"({pct:4.0f}%) [{bar}] {speed:6.1f} MB/s")
    else:
        line = f"\r  {name:16s} {mb:7.1f} MB ... {speed:6.1f} MB/s"
    sys.stderr.write(line)
    sys.stderr.flush()


def fetch_one(name: str, info: dict) -> Path:
    local = Path(info["local"])
    if local.exists():
        got = sha256_file(local)
        if got == info["sha256"]:
            print(f"verified, reuse: {local}")
            return local
        print(f"hash mismatch — re-fetching: {local} (got {got})", file=sys.stderr)
        local.unlink(missing_ok=True)
    print(f"fetching ({name}) {info['url']}")
    tmp = local.with_suffix(local.suffix + ".partial")
    request = urllib.request.Request(
        info["url"], headers={"User-Agent": "pianist-fly-setup/1.0"})
    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            t0 = time.monotonic()
            last = 0.0
            with open(tmp, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last >= 0.1 or (total and done >= total):
                        last = now
                        _progress(name, done, total, t0)
            _progress(name, done, total, t0)
        sys.stderr.write("\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("\n")
        print(f"fetch failed for {name}: {e}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        raise
    got = sha256_file(tmp)
    if got != info["sha256"]:
        print(f"hash mismatch — refusing {name} (got {got} "
              f"want {info['sha256']})", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        raise SystemExit(2)
    tmp.replace(local)
    print(f"verified: {local}")
    return local


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/graph.npz"))
    ap.add_argument("--no-build", action="store_true",
                    help="only fetch and verify feathers; skip the compile")
    a = ap.parse_args()

    local = {}
    for name, info in CANONICAL.items():
        local[name] = fetch_one(name, info)

    if a.no_build:
        return 0
    ROOT = Path(__file__).resolve().parents[1]
    build = ROOT / "scripts" / "build_graph_npz.py"
    cmd = [sys.executable, str(build),
           "--annotations", str(local["annotations"]),
           "--neurotransmitters", str(local["neurotransmitters"]),
           "--edges", str(local["edges"]),
           "--out", str(a.out)]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())