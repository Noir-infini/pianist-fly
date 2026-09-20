"""NativeBrain v2 — vectorized leaky integrate-and-fire over the MaleCNS
connectome. Standalone: no doom import, no doom repo, no decoder-as-conductor.

Dynamics match the documented MaleCNS LIF exactly (rest -52 mV, threshold
-45 mV, membrane tau 20 ms, conductance tau 5 ms, synaptic delay 1.8 ms,
refractory 2.2 ms, tonic drive on lamina/retina/sugar, spike delivery through a
delayed ring queue). The reference kernel is an audit-friendly simulator the
author used; THIS class is a self-contained numpy port of the same math: same
driving, same thresholds, same delays — but no external engine is linked,
compiled, or imported.

Data is our own vendored connectome:
  data/graph.npz                 (ptr, post, weight, ids, retina, uv, lamina,
                                  sugar, superclass)
  data/mcns_annotations.feather  (somaNeuromere/somaSide for the vnc_motor pools)

Every spike this class produces is `predicted` (a simulated firing rate), never
a claim of measuring a real fly.

Why it is fast enough to run live:
  * Evolving the membrane at every dt=0.1 ms substep is exactly equivalent to
    the reference kernel's lazy closed-form over the same constant drive — no
    neuron is ever skipped in a way that changes a spike.
  * Spikes are delivered through the same delayed ring queue; the expensive
    part (weighted scatter onto postsynaptic targets) is one sparse gather +
    np.bincount over the edges of the neurons that actually fired this substep
    — O(edges fired), not O(all 25.6M edges).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data/graph.npz"
FEATHER = ROOT / "data/mcns_annotations.feather"


class NativeBrain:
    """Vectorized MaleCNS LIF. The whole brain lives in a handful of float
    arrays; one `step` evolves `round(duration_ms/0.1)` substeps at dt=0.1 ms.
    """

    DT = 0.1          # ms (the only dt the documented kernel supports)

    def __init__(self, graph_path: Path = GRAPH, dt: float = DT):
        if dt != self.DT:
            raise ValueError("This engine supports only dt=0.1 ms.")
        g = np.load(graph_path, allow_pickle=True)
        for k in ("ptr", "post", "weight", "ids", "retina", "uv", "lamina",
                  "sugar", "superclass"):
            if k not in g:
                raise ValueError(f"connectome missing key: {k}")
            setattr(self, k, np.asarray(g[k]))
        n = len(self.ids)
        for k, dtype in [("ptr", np.int64), ("post", np.int32),
                         ("weight", np.float32), ("ids", np.int64)]:
            if getattr(self, k).ndim != 1 or getattr(self, k).dtype != dtype:
                raise ValueError(f"Invalid native graph array: {k}")
        if self.ptr.shape != (n + 1,) or self.ptr[0] != 0 or \
                self.ptr[-1] != len(self.post) or \
                np.any(np.diff(self.ptr) < 0) or \
                len(self.weight) != len(self.post):
            raise ValueError("Invalid CSR graph")
        if not np.isfinite(self.weight).all():
            raise ValueError("Non-finite synaptic weight")
        for x in (self.post, self.retina, self.lamina, self.sugar):
            if np.any(x < 0) or np.any(x >= n):
                raise ValueError("Graph index out of bounds")
        if self.uv.shape != (len(self.retina), 2) or \
                not np.isfinite(self.uv).all() or np.any(self.uv < 0) or \
                np.any(self.uv > 1):
            raise ValueError("Invalid receptor UV coordinates")

        self.dt = dt
        self.n = n
        self.delay = int(round(1.8 / dt))     # synaptic delay (substeps)
        self.rfc = int(round(2.2 / dt))       # refractory period (substeps)
        self.slots = self.delay + 1

        # Membrane / synaptic state (documented parameter choices).
        self.v = np.full(n, -52.0, dtype=np.float32)       # mV
        self.g = np.zeros(n, dtype=np.float32)             # conductance bump
        self.drive = np.zeros(n, dtype=np.float32)         # tonic current
        self.refractory = np.zeros(n, dtype=np.int16)      # substeps remaining
        self.luminance = np.zeros(len(self.retina), dtype=np.float32)
        self.counts = np.zeros(n, dtype=np.int32)          # per-step spikes

        # Ring of spike queues: slot s holds neurons to DELIVER while the
        # clock is at s (they fired `delay` substeps earlier).
        self.queue = [[] for _ in range(self.slots)]
        self.clock = 0                     # clock substeps
        self.total_spikes = 0
        self.sim_ms = 0.0

        # Decay per substep: membrane tau 20 ms, conductance tau 5 ms.
        self.a1 = math.exp(-dt / 20.0)
        self.b1 = math.exp(-dt / 5.0)

    # -- stimulus plumbing ----------------------------------------------------
    def _make_drive(self, lamina_bias, sugar, extra_drive):
        drive = self.drive
        drive.fill(0.0)
        drive[self.lamina] = lamina_bias
        drive[self.retina] = 30.0 * self.luminance / (0.02 + self.luminance)
        if sugar:
            drive[self.sugar] = 30.0
        if extra_drive is not None:
            if isinstance(extra_drive, dict):
                items = extra_drive.items()
            elif isinstance(extra_drive, np.ndarray):
                drive += extra_drive
                items = None
            elif isinstance(extra_drive, (tuple, list)):
                if len(extra_drive) == 2 and not isinstance(extra_drive[0], tuple):
                    items = [extra_drive]
                else:
                    items = extra_drive
            else:
                items = None
            if items is not None:
                for idx, val in items:
                    drive[idx] = val
        return drive

    # -- the exact per-substep loop --------------------------------------------
    def _run(self, steps):
        """Evolve the brain for `steps` substeps of dt, filling self.counts."""
        n = self.n
        v, g = self.v, self.g
        drive = self.drive
        ref = self.refractory
        counts = self.counts
        a1, b1 = self.a1, self.b1
        ptr, post, weight = self.ptr, self.post, self.weight
        dly, rfc, slots = self.delay, self.rfc, self.slots
        queue = self.queue
        clock = self.clock

        for _ in range(steps):
            slot = clock % slots
            future = (clock + dly) % slots

            # Decrement refractory; a neuron was frozen only if it still had
            # >1 substeps left before this one (see kernel evolve()).
            integ = ref <= 1
            ref -= 1
            ref[ref < 0] = 0

            # Membrane evolution (exact closed form at d=1 substep).
            if integ.any():
                vk = v[integ]
                v[integ] = (-52.0 + (vk + 52.0) * a1
                            + drive[integ] * (1.0 - a1)
                            + g[integ] * ((a1 - b1) / 3.0))
                np.multiply(g, b1, out=g, where=integ)  # zeroed above, okay

            # Fire & immediate reset (v=-52, g=0, refract=rfc).
            fires = np.flatnonzero((ref == 0) & (v > -45.0))
            if fires.size:
                counts[fires] += 1
                queue[future].append(fires)
                v[fires] = -52.0
                g[fires] = 0.0
                ref[fires] = rfc

            # Deliver the spikes that fired `delay` substeps ago: a weighted
            # scatter onto their postsynaptic targets (skipping targets that
            # are themselves refractory this substep).
            delivered = queue[slot]
            if delivered:
                queue[slot] = []
                f = np.concatenate([np.asarray(x, dtype=np.int64)
                                    for x in delivered])
                starts = ptr[f]
                lens = ptr[f + 1] - starts
                m = int(lens.sum())
                if m:
                    local = np.arange(m, dtype=np.int64) - np.repeat(
                        np.cumsum(lens) - lens, lens)
                    eid = np.repeat(starts, lens) + local
                    bump = np.bincount(post[eid], weights=weight[eid],
                                       minlength=n)
                    g += bump * (ref == 0)

            clock += 1
        self.clock = clock

    # -- public API -------------------------------------------------------------
    def step(self, luminance, duration_ms, sugar=False, lamina_bias=12.0,
             extra_drive=None):
        """One control tick. `luminance` is a finite (3335,) map of receptor
        values in 0..1 (same shape as uv). Returns (spike_counts, wall_ms)."""
        lum = np.asarray(luminance, dtype=np.float32)
        if lum.shape != (len(self.retina),):
            raise ValueError("Luminance must have one value per mapped receptor")
        if not np.all(np.isfinite(lum)):
            raise ValueError("A finite luminance sample is required per receptor")
        if not math.isfinite(duration_ms) or not math.isfinite(lamina_bias):
            raise ValueError("Finite duration and current required")

        steps = int(round(duration_ms / self.dt))
        if steps < 1:
            raise ValueError("Duration too short")

        # Discrete low-pass on the optical input (documented, not calibrated
        # phototransduction).
        alpha = 1.0 - math.exp(-steps * self.dt / 10.0)
        self.luminance += alpha * (np.clip(lum, 0.0, 1.0) - self.luminance)

        self._make_drive(lamina_bias, bool(sugar), extra_drive)
        self.counts.fill(0)
        t0 = time.perf_counter()
        self._run(steps)
        wall = time.perf_counter() - t0
        self.total_spikes += int(self.counts.sum())
        self.sim_ms += steps * self.dt
        return self.counts.copy(), wall


def vendored():  # provenance, honest
    return {"source": {"kind": "predicted",
                       "name": "NativeBrain v2 (vectorized MaleCNS LIF, numpy)",
                       "path": str(GRAPH)}}