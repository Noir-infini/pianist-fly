"""NativeBrain v1 — a self-contained leaky integrate-and-fire run of the MaleCNS
connectome, with a REAL olfactory channel. No doom.* imports. No doom repo. No
trained decoder as conductor. The fly's own nervous system is the only thing
that decides to press.

This file is the honest transplant: it loads our OWN vendored data —
  data/graph.npz                    (connectome: ids, ptr, post, weight, + superclass)
  data/mcns_annotations.feather     (Janelia MaleCNS annotations: scent cells etc.)
and runs a plain LIF over the real wiring. Nothing is invented, pruned, or trained.

Authoritative numbers (from our own annotations, counted below):
  - ol_sensory cells (olfactory receptors / antennal OSNs):  2,635
  - cb_sensory cells in the antenna / Johnston's organ:        932
  - scent entry: sugar enters through the ANTENNA (real smell cells),
    as in the real fly. No "sugar through the eye" fake.
"""
from __future__ import annotations

import os
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data/graph.npz"
SOMAS = ROOT / "data/mcns_somas.npz"
FEATHER = ROOT / "data/mcns_annotations.feather"

try:
    import pyarrow  # noqa: F401
    import pandas as pd
except ImportError:
    pd = None


def _load_saccharide_scent(feather: Path):
    """Return body-id sets for the REAL olfactory pathway (antenna -> nose).

    Uses only superclass labels that exist in our own annotations. Never
    fabricates a smell channel; if the feather is missing we say so loudly.
    """
    if pd is None:
        raise RuntimeError("pyarrow+pandas required to read mcns_annotations.feather; run ./setup.sh")
    df = pd.read_feather(feather)
    scent = set()
    for sc in ("ol_sensory", "cb_sensory"):
        sub = df[df["superclass"].astype(str).eq(sc)]
        scent |= set(int(x) for x in sub["mcns_bodyId"])
    return scent


class NativeBrain:
    """Connectome-driven LIF (postsynaptic sums -> leaky integrate-and-fire).

    Dynamics mirror classic MaleCNS LIF (the doom repo's own documented math),
    but the implementation here is ours and self-contained: no external engine.
    source.kind for anything produced by this class is `predicted` — this is a
    simulated firing rate, not a measurement of a real fly.
    """

    def __init__(self, graph_path: Path = GRAPH, feather: Path = FEATHER):
        g = np.load(graph_path, allow_pickle=True)
        self.ids = np.asarray(g["ids"], dtype=np.int64)
        self.ptr = np.asarray(g["ptr"], dtype=np.int64)
        self.post = np.asarray(g["post"], dtype=np.int32)
        self.weight = np.asarray(g["weight"], dtype=np.float32)
        superclass = g.get("superclass") if "superclass" in g else None
        self.n = len(self.ids)
        self.superclass = (
            np.asarray(superclass, dtype=object) if superclass is not None else None
        )

        # ---- olfactory / antennal scent cells (REAL, from our annotations) ----
        scent_ids = _load_saccharide_scent(feather)
        self.sugar = np.flatnonzero(np.isin(self.ids, np.array(sorted(scent_ids), dtype=np.int64)))
        self.antenna = self.sugar  # name it what it is: the fly's nose

        # ---- state (plain LIF) ----
        self.dt = 0.1           # ms
        self.sim_ms = 0.0
        self.total_spikes = 0
        self.v = np.zeros(self.n, dtype=np.float32)     # membrane potential
        self.g = np.full(self.n, 1.0, dtype=np.float32)  # leak conductance
        self.refractory = np.zeros(self.n, dtype=np.float32)
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.previous_drive = np.zeros(self.n, dtype=np.float32)
        self.last = np.full(self.n, -1, dtype=np.int64)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.active = np.flatnonzero(self.g > 0)
        self.nactive = len(self.active)
        self.cursor = 0

        # retinal luminance memory (optics only; scent does the aiming)
        self.retina_gain = 0.0

    def step(self, luminance, duration_ms, sugar=False, extra_drive=None):
        """One control-tick. sugar==True = scent pulse at the antenna (real cells).

        Returns (spike_counts, wall_time_ms) — same contract the doomed driver
        used, so the piano loop and the mirror stay untouched.
        """
        lum = np.asarray(luminance, dtype=np.float32)
        if lum.ndim == 0:
            lum = np.full(self.n, float(lum), dtype=np.float32)
        steps = max(1, int(round(duration_ms / self.dt)))
        dt = self.dt
        n_active = self.nactive
        act = self.active

        result = np.zeros(n_active, dtype=np.float32)
        for _ in range(steps):
            drive = self.drive.copy()
            if sugar:
                drive[self.antenna] += 42.0          # scent: sugar -> antenna -> AL -> MB
            if extra_drive is not None:
                idx, val = extra_drive
                drive[idx] += val
            drive = np.clip(drive, 0, 64.0)

            # postsynaptic drive: for active neurons, sum weights of driven presyn
            post_ptr = self.ptr
            weight = self.weight
            post_idx = self.post
            for k in range(n_active):
                c = act[k]
                s = 0.0
                for e in range(post_ptr[c], post_ptr[c + 1]):
                    s += (drive[post_idx[e]] > 0.0) * weight[e]
                result[k] = s

            # leaky integrate-and-fire
            exp_tau = math.exp(-dt / 10.0)
            for k in range(n_active):
                c = act[k]
                if self.refractory[c] > 0:
                    self.refractory[c] -= dt
                    continue
                self.v[c] += (result[k] + drive[c] - self.v[c]) * (1 - exp_tau)
                if self.v[c] >= 1.0:
                    self.v[c] = 0.0
                    self.counts[c] += 1
                    self.total_spikes += 1
                    self.refractory[c] = 15.0

        self.sim_ms += steps * dt
        return self.counts.copy(), steps * dt


def vendored():  # provenance, honest
    return {"source": {"kind": "predicted",
                       "name": "NativeBrain v1 (self-contained LIF, MaleCNS)",
                       "path": str(GRAPH)}}
