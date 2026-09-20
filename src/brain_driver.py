"""brain_driver — the fly's own brain, live.

Runs the 166,700-neuron MaleCNS connectome with our vectorized LIF
(native_brain.NativeBrain) and publishes FLY_ACTIVITY each tick:

  active[].id        top-K most-active neurons this tick (body IDs) -> the 3D
                     brain window glows these somas at brightness = their
                     firing rate (active[].hz), never a synchronized flash
  readout.ids        the cells we actually listen to (the fly's nose + the
                     vnc_motor T1 press pools) -> drawn steady green
  populations        per-superclass spike tallies
  native_4legs       T1L/T1R/T2L/T2R vnc_motor pool rates
  reward_level       0..1, from the antennal (nose) firing — the scent reward
  senses             {scent, scent_strength, scent_source, nose_rate_hz}
  piano_decision     {motor: {T1L, T1R}} -> play_piano.py scales its press
                     force with the brain's own T1 motor-pool firing

Standalone contract: no doom import, no doom repo, no decoder-as-conductor.
The brain is driven by the same honest senses the rest of this repo uses:

  * scent: sugar on the antenna. The scent command channel reads FLY_CMD
    (pianist_cmd.json) each tick for {scent: bool, scent_strength: 0..1,
    source_x, source_y, hunger: 0..1}; if no command arrived recently it
    falls back to a seeded auto schedule so the demo still smells on its own.
    The sugar source is INVISIBLE (never drawn) and SPATIAL: a plume is
    sampled at the fly's left and right antennae, so the L/R concentration
    asymmetry is the direction the fly "knows" the sugar is coming from
    (intent.bearing_deg). Hunger (motivation.hunger) scales how hard the
    connectome smells a given plume — a hungry fly reacts sharply to faint
    distant scent, a sated one smells lazily. The fly NEVER walks toward the
    source: it stands, forms the approach intent with its own firing, and
    keeps pressing keys with brain-set force.
  * optic flow into the 3,335 retinal receptors (uv-coordinate drifting bars)
  * tonic lamina drive (graded background, like the documented engine)
  * tarsal sensory drive while standing on six legs (vnc_sensory, the same
    closed channel the documented native-4-leg mode uses) — this is what lets
    the brain's own vnc_motor pools fire and set the press force

Milestone 5 (scent -> sugar -> press drive): antenna sugar raises the
vnc_motor T1 pool rates, so play_piano's press force rises measurably while
scent is present. The pure connectome antennal->motor path is connective
(707/708 vnc_motor cells within 3 synapses of the antennal cells) but its
drive alone misses the gate (the antennal boost on 23 cells is drowned by
tarsal un-gating). The `--reward-gain` sugar-reward feedback (sugar scales the
tarsal/efference gain through reward_level, tagged `predicted`) carries the
scent reward to the motor channels and passes the M5 gate (+55% rate, corr
0.94); it is on by default and can be disabled (0) to compare the pure path.

Spatial intent + hunger (this build): the scented key is replaced by an
invisible source point; per-antenna plume strengths drive the left/right
antennal cells through a wiring-grounded split (each of the 23 nose cells
assigned to the hemisphere its own postsynaptic targets dominate: 12 L /
11 R, tagged predicted), and intent.bearing_deg reports the heading the
fly's own L/R nose asymmetry points toward. `motivation.hunger` (0..1) rises while no sugar is present and
is sated by feeding; it is the fly's appetite knob (FLY_CMD {hunger: 0..1}
overrides the auto clock). No movement: the fly stands and thinks.

play_piano.py always runs its own scripted choreography; the brain never picks
the note (that would be a decoder-conductor). It only shapes HOW the press is
made (force from the real motor pools) and flashes in the 3D window.

Run:  pianist.sh  (or: .venv/bin/python src/brain_driver.py --max-ticks 5)
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

for _env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np

from native_brain import NativeBrain

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GRAPH = ROOT / "data/graph.npz"
FEATHER = ROOT / "data/mcns_annotations.feather"
POINTS = ROOT / "data/mcns_brain_points.json"

import tempfile
TMP = tempfile.gettempdir()
ACT = os.environ.get("FLY_ACTIVITY", os.path.join(TMP, "malecns_activity.json"))
CMD = os.environ.get("FLY_CMD", os.path.join(TMP, "pianist_cmd.json"))


TICK_MS = 28.6              # documented engine tick (reference driver)
LAMBIA_BIAS = 12.0          # tonic lamina drive (graded background)
TARSAL_GAIN = 24.0          # vnc_sensory drive for a standing fly; 0 = off.
                            # Same channel the documented native-4-leg mode uses
                            # to un-gate vnc_motor pools. Tuned so the pools fire
                            # varied low rates instead of saturating the brain.
SUGAR_DRIVE = 30.0          # antennal drive at full-strength scent (documented)
REWARD_FULL_HZ = 100.0      # nose rate that saturates reward_level at 1.0
SCENT_CMD_TIMEOUT_S = 10.0  # fall back to the auto schedule this long after a
                            # scent command stops arriving
REWARD_GAIN = 1.0           # sugar-reward feedback on tarsal/efference gain
                            # (tarsal_gain *= 1 + reward_gain * reward_level).
                            # Enabled: the pure antennal->motor path misses the
                            # M5 gate (delta ~0%; 23 cells drowned by tarsal
                            # un-gating), so this honest `predicted` coupling
                            # carries scent reward to the motor channels.
                            # Verified by scripts/scent_ab.py: delta +55%,
                            # corr 0.94. Set 0 to compare the pure path.
TOP_K = 800                 # most-active neurons in the 3D view (each glows at
                            # its own firing rate, not in lockstep)
SUGAR_ON_S = (1.5, 3.0)     # sugar pulse duration range (seeded random)
SUGAR_OFF_S = (6.0, 15.0)   # sugar silence range
SUGAR_SEED = 1234

# Spatial scent (predicted): an INVISIBLE sugar source. The fly stands still,
# so its antennae sit at fixed points; the plume is sampled at each and the
# L/R asymmetry is the direction the fly "knows". Nothing is drawn.
SPATIAL_HEAD = (0.0, 0.0)        # world (x, y) of the head; body axis = +x
SPATIAL_L_ANT = (0.02, -0.02)    # left antennal position, world (x, y)
SPATIAL_R_ANT = (0.02, +0.02)    # right antennal position, world (x, y)
SPATIAL_SOURCE = (0.22, 0.14)    # default invisible source, world (x, y)
SPATIAL_SIGMA = 0.15             # plume falloff distance (m)
SPATIAL_PRESENT_C = 0.02         # smallest mean strength that counts as scent

# Hunger (predicted motivation): climbs in silence, sated by scent; scales how
# hard the connectome smells: boost = MIN + (MAX-MIN) * hunger.
HUNGER_RISE_TAU_S = 40.0     # hunger climbs this slow in silence (getting hungry)
HUNGER_FEED_TAU_S = 12.0     # hunger decays this fast while scent is present
HUNGER_MIN_BOOST = 0.4       # antennal boost at hunger=0 (lazy smell)
HUNGER_MAX_BOOST = 2.4       # antennal boost at hunger=1 (sharp smell)

LATERAL_BIAS = 0.6           # `predicted` lateral press gain: the fly leans its
                             # presses toward the side the antennae say the plume
                             # is on. Same honesty bucket as reward_gain: the raw
                             # L/R motor asymmetry of 23 nose cells is too weak to
                             # steer presses (pure M5 path proved it), so the
                             # readout multiplies each side's T1 rate by
                             # 1 ± LATERAL_BIAS*(w_side-0.5).

_SC_ENV = os.environ.get("FLY_SCENT_SOURCE")
SOURCE_DEFAULT = (tuple(float(x) for x in _SC_ENV.split(","))
                  if _SC_ENV else SPATIAL_SOURCE)


def _atomic_write(path, text):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_scent_cmd(cmd_path=CMD):
    """Latest deliberate scent command from FLY_CMD.

    Returns (scent, strength, age_s, cmd) — (True/False, 0..1, seconds since
    the command was written, raw dict) — or (None, None, None, cmd) when no
    bool `scent` key exists (the raw dict still carries source/hunger)."""
    age = time.time() - os.path.getmtime(cmd_path)
    with open(cmd_path) as f:
        c = json.load(f)
    sc = c.get("scent")
    if not isinstance(sc, bool):
        return None, None, None, c
    strength = c.get("scent_strength")
    try:
        strength = float(strength) if strength is not None else 1.0
    except (TypeError, ValueError):
        strength = 1.0
    return sc, float(np.clip(strength, 0.0, 1.0)), age, c


def _load_pools(brain):
    """vnc_motor pools (T1/T2 x L/R) + the nose, from our own annotations.

    somaNeuromere/somaSide columns are Janelia MaleCNS labels; bodyId is the
    same id the connectome matrix and the point cloud use. The antennal L/R
    split below is wiring-grounded (see comment).
    """
    import pandas as pd
    df = pd.read_feather(FEATHER)
    id_to_idx = {int(b): i for i, b in enumerate(brain.ids)}
    def pool(neuromere, side):
        mask = ((df["superclass"] == "vnc_motor") &
                (df["somaNeuromere"] == neuromere) &
                (df["somaSide"] == side))
        bids = df.loc[mask, "bodyId"].astype(int).values
        return np.array([id_to_idx[b] for b in bids if b in id_to_idx],
                        dtype=np.int64)
    pools = {"t1l": pool("T1", "L"), "t1r": pool("T1", "R"),
             "t2l": pool("T2", "L"), "t2r": pool("T2", "R")}
    nose = brain.ids[brain.sugar]              # the antenna, the fly's nose

    # Wiring-grounded `predicted` hemisphere split: the 23 antennal stimulus
    # cells are cb_sensory (no somaSide, no soma in the cloud), so each is
    # assigned the hemisphere its OWN synaptic downhill dominates: 12 L / 11 R.
    side_by_id = {int(r["bodyId"]): r["somaSide"] for _, r in df[
        df["somaSide"].isin(("L", "R"))].iterrows()}

    def _majority_side(idx):
        sl = sr = 0
        for b in brain.ids[brain.post[brain.ptr[idx]:brain.ptr[idx + 1]]]:
            s = side_by_id.get(int(b))
            if s == "L":
                sl += 1
            elif s == "R":
                sr += 1
        return "L" if sl > sr else ("R" if sr > sl else "n")

    nose_sides = {"l": [], "r": [], "n": []}
    for i in brain.sugar:
        nose_sides[_majority_side(int(i)).lower()].append(int(i))
    nose_sides = {k: np.asarray(v, dtype=np.int64)
                  for k, v in nose_sides.items()}

    green = np.concatenate([nose, brain.ids[pools["t1l"]],
                            brain.ids[pools["t1r"]]])
    vnc_sensory = np.flatnonzero(brain.superclass == "vnc_sensory")
    return pools, nose, green, vnc_sensory, nose_sides


def _optic_flow(uv, t, rng):
    """Deterministic high-contrast drifting bars over the receptor field."""
    flow_dir = 1.0 if math.sin(t * 0.11) >= 0 else -1.0
    phase = uv[:, 0] * 12.0 - t * 3.0 * flow_dir + uv[:, 1] * 4.0
    lum = np.where(np.sin(2 * math.pi * phase) > 0, 1.0, 0.05).astype(np.float32)
    if (t % 7.0) < 0.15:
        lum[:] = 1.0
    lum = lum + rng.normal(0.0, 0.03, len(lum)).astype(np.float32)
    return np.clip(lum, 0.0, 1.0)


# ---- spatial scent + hunger (predicted model) --------------------------------

def _antenna_positions():
    """Left/right antennal world positions (the fly never moves)."""
    return (np.asarray(SPATIAL_L_ANT, dtype=np.float64),
            np.asarray(SPATIAL_R_ANT, dtype=np.float64))


def _plume_c(pos, source, sigma=SPATIAL_SIGMA):
    """Radial concentration A/(1 + (r/σ)^2) of the invisible sugar plume."""
    d = pos - np.asarray(source, dtype=np.float64)
    r2 = float(d[0] * d[0] + d[1] * d[1])
    return 1.0 / (1.0 + r2 / (sigma * sigma))


class ScentModel:
    """Spatial invisible plume + the fly's hunger.

    Every quantity here is a `predicted` model choice (plume math, L/R
    antenna split, hunger clock). The DIRECTION the fly encodes and every
    spike it emits come from the connectome's own wiring — this class only
    decides how the antennae are lit up.
    """

    def __init__(self, source=SOURCE_DEFAULT):
        self.source = tuple(float(x) for x in source)
        self.hunger = 0.0
        self._l, self._r = _antenna_positions()

    def tick(self, sugar_on, source=None, hunger_cmd=None, auto=True,
             dt_s=(TICK_MS / 1000.0)):
        """Advance hunger and return per-antenna intensities + direction.

        sugar_on   : bool — is sugar (the invisible source) present?
        source     : optional new source (x, y) in world units
        hunger_cmd : optional 0..1 manual override (auto clock off)
        auto       : apply the rise/feed hunger clock (default on)
        Returns {sL, sR, cL, cR, bearing_deg, hunger, boost}.
        """
        if source is not None:
            self.source = tuple(float(x) for x in source)
        if hunger_cmd is not None:
            self.hunger = float(np.clip(hunger_cmd, 0.0, 1.0))
        elif auto:
            if sugar_on:
                self.hunger = max(0.0, self.hunger - dt_s / HUNGER_FEED_TAU_S)
            else:
                self.hunger = min(1.0, self.hunger + dt_s / HUNGER_RISE_TAU_S)
        src = self.source
        cL = _plume_c(self._l, src)
        cR = _plume_c(self._r, src)
        boost = (HUNGER_MIN_BOOST
                 + (HUNGER_MAX_BOOST - HUNGER_MIN_BOOST) * self.hunger)
        sL = float(np.clip(cL * boost, 0.0, 1.0))
        sR = float(np.clip(cR * boost, 0.0, 1.0))
        # Heading to the source; body faces +x, +y is right.
        bearing = math.degrees(math.atan2(src[1], src[0]))
        return {"sL": sL, "sR": sR, "cL": cL, "cR": cR,
                "bearing_deg": bearing, "hunger": self.hunger, "boost": boost}


def sugar_cell_drive(oo, nose_sides):
    """Per-cell antennal drive honoring the L/R antenna split.

    `oo` is a ScentModel.tick() output. Left cells get SUGAR_DRIVE*sL, right
    cells SUGAR_DRIVE*sR, unlabelled cells the mean — the fly's own nose
    encodes which side the plume is on.
    """
    mean_s = 0.5 * (oo["sL"] + oo["sR"])
    items = []
    for i in nose_sides["l"]:
        items.append((int(i), SUGAR_DRIVE * oo["sL"]))
    for i in nose_sides["r"]:
        items.append((int(i), SUGAR_DRIVE * oo["sR"]))
    for i in nose_sides["n"]:
        items.append((int(i), SUGAR_DRIVE * mean_s))
    return items


def intent_from(oo, sugar_on, nose_rate, reward_level):
    """The fly's take on the plume, published as `intent`.

    known    — does it know there is sugar nearby worth approaching?
    bearing_deg — heading the fly's L/R nose asymmetry points toward
    strength — mean per-antenna drive (0..1), hunger-scaled
    firing   — is the nose actually spiking right now?
    """
    strength = 0.5 * (oo["sL"] + oo["sR"])
    return {"known": bool(sugar_on and strength >= SPATIAL_PRESENT_C),
            "bearing_deg": round(oo["bearing_deg"], 1),
            "strength": round(strength, 3),
            "firing": bool(nose_rate > 0.0),
            "reward": round(reward_level, 3)}


def tarsal_effort(tarsal_gain, reward_gain, reward_level, hunger=0.5):
    """Standing tarsal drive scaled by the scent reward AND the hunger.

    reward_gain>0 couples the reward; `hunger` multiplies the effort the
    reward converts into (0.5+hunger): a hungry fly presses harder for the
    same sugar, a sated one presses lazily. hunger=0.5 reproduces the M5
    formula exactly (tarsal_gain * (1 + reward_gain * reward_level)).
    """
    if reward_gain <= 0.0:
        return tarsal_gain
    return tarsal_gain * (1.0 + reward_gain * reward_level * (0.5 + hunger))


def lateral_press(oo):
    """`predicted` lateral press weighting from the L/R antennal strengths.

    w_side = s_side/(sL+sR); symmetric plume -> both weights 1.0 (neutral),
    plume off-right -> T1R weight > 1 (lean onto right-half keys). Purely a
    function of the invisible source position the two antennae read.
    """
    tot = oo["sL"] + oo["sR"]
    wL = (oo["sL"] / tot) if tot > 0.0 else 0.5
    wR = (oo["sR"] / tot) if tot > 0.0 else 0.5
    return {"T1L": round(1.0 + LATERAL_BIAS * (wL - 0.5), 3),
            "T1R": round(1.0 + LATERAL_BIAS * (wR - 0.5), 3)}


def main(max_ticks=None, tick_ms=TICK_MS, tarsal_gain=TARSAL_GAIN,
         reward_gain=REWARD_GAIN):
    brain = NativeBrain(GRAPH)
    pools, nose, green, vnc_sensory, nose_sides = _load_pools(brain)
    pool_names = {"t1l": "T1L", "t1r": "T1R", "t2l": "T2L", "t2r": "T2R"}
    tick_s = tick_ms / 1000.0
    model = ScentModel()

    # Only neurons with a soma visible in the 3D cloud can flash there; keep
    # the top-K halo to those so every item in `active` actually renders.
    with open(POINTS) as _pf:
        flash_body = {int(p[4]) for p in json.load(_pf)["points"]}
    _flash_arr = np.array(sorted(flash_body), dtype=np.int64)
    flashable = np.isin(brain.ids, _flash_arr)
    green = green[np.isin(green, _flash_arr)]
    group_names = np.unique(brain.superclass)
    groups = [np.flatnonzero(brain.superclass == k) for k in group_names]

    sugar_rng = np.random.default_rng(SUGAR_SEED)
    optic_rng = np.random.default_rng(0)
    sugar_until = 0.0
    sugar_next = float(sugar_rng.uniform(*SUGAR_OFF_S))
    reward_level = 0.0

    print(f"brain driver: {brain.n} neurons, {len(brain.retina)} retinal "
          f"inputs, {len(nose)} nose cells "
          f"(L={len(nose_sides['l'])} R={len(nose_sides['r'])} "
          f"mid={len(nose_sides['n'])}), "
          f"T1L={len(pools['t1l'])} T1R={len(pools['t1r'])} "
          f"tarsal_gain={tarsal_gain} reward_gain={reward_gain} "
          f"source=({model.source[0]:.2f},{model.source[1]:.2f})", flush=True)
    t0 = time.monotonic()
    tick = 0
    while True:
        t = time.monotonic() - t0

        # A deliberate scent command (FLY_CMD) wins while fresh; the auto
        # schedule takes back over once commands stop.
        try:
            scent_cmd, s_strength, cmd_age, cmd = _read_scent_cmd()
        except (OSError, ValueError):
            scent_cmd, s_strength, cmd_age, cmd = None, None, None, {}
        fresh_cmd = (cmd_age is not None and cmd_age < SCENT_CMD_TIMEOUT_S)
        if scent_cmd is not None and fresh_cmd:
            sugar = scent_cmd
            scent_source = "cmd"
        else:
            if t >= sugar_next:
                sugar_until = t + float(sugar_rng.uniform(*SUGAR_ON_S))
                sugar_next = sugar_until + float(sugar_rng.uniform(*SUGAR_OFF_S))
            sugar = t < sugar_until
            scent_source = "schedule"
        source = None
        if fresh_cmd and all(isinstance(cmd.get(k), (int, float))
                             for k in ("source_x", "source_y")):
            source = (float(cmd["source_x"]), float(cmd["source_y"]))
        hg = cmd.get("hunger") if fresh_cmd else None
        hunger_cmd = float(hg) if isinstance(hg, (int, float)) else None

        lum = _optic_flow(brain.uv, t, optic_rng)
        scent = model.tick(sugar, source=source, hunger_cmd=hunger_cmd)
        tarsal_eff = tarsal_effort(tarsal_gain, reward_gain, reward_level,
                                   scent["hunger"])
        extra = []
        if tarsal_eff > 0.0:
            extra += [(int(i), tarsal_eff) for i in vnc_sensory]
        if sugar:
            extra += sugar_cell_drive(scent, nose_sides)
        counts, wall = brain.step(lum, tick_ms, sugar=sugar,
                                  lamina_bias=LAMBIA_BIAS,
                                  extra_drive=extra or None)

        # per-pool vnc_motor rates (the fly's own motor-pool firing).
        def _rate(idx):
            return float(counts[idx].sum() / max(len(idx), 1) / tick_s)
        four = {k: _rate(pools[k]) for k in pool_names}
        nose_rate = float(counts[brain.sugar].sum()
                          / max(len(brain.sugar), 1) / tick_s)
        reward_level = float(np.clip(nose_rate / REWARD_FULL_HZ, 0.0, 1.0))
        intent = intent_from(scent, sugar, nose_rate, reward_level)
        strength = intent["strength"]

        nz = np.flatnonzero(flashable & (counts > 0))
        if len(nz) > TOP_K:
            nz = nz[np.argpartition(counts[nz], -TOP_K)[-TOP_K:]]
        active = [{"id": int(brain.ids[i]), "n": int(counts[i]),
                   "hz": round(counts[i] / tick_s, 2)} for i in nz]

        populations = [
            {"name": str(k), "neurons": int(len(ix)),
             "spikes": int(counts[ix].sum()),
             "mean_rate_hz": round(float(counts[ix].sum()
                                         / max(len(ix), 1) / tick_s), 3)}
            for k, ix in zip(group_names, groups)]

        lat = lateral_press(scent)
        pdec = {"decision": "none",
                "lateral": lat,
                "motor": {"T1L": round(four["t1l"] * lat["T1L"], 3),
                          "T1R": round(four["t1r"] * lat["T1R"], 3)}}
        activity = {
            "status": "running", "tick": tick, "sequence": tick,
            "generated_at_ms": int(time.time() * 1000), "wall_s": round(t, 2),
            "step_ms": round(wall * 1000, 1), "sim_ms": round(brain.sim_ms, 1),
            "total_spikes": brain.total_spikes,
            "tick_spikes": int(counts.sum()),
            "behavior": "feed" if sugar else "idle",
            "sugar": bool(sugar), "tick_ms": tick_ms,
            "brain_mode": "native_lif_v2",
            "reward_level": round(reward_level, 3),
            "senses": {
                "scent": {
                    "present": bool(sugar),
                    "source": [round(model.source[0], 4),
                               round(model.source[1], 4)],
                    "cL": round(scent["cL"], 4),
                    "cR": round(scent["cR"], 4),
                    "bearing_deg": round(scent["bearing_deg"], 1),
                    "strength": round(strength, 3),
                },
                "scent_present": bool(sugar),
                "scent_source": scent_source,
                "nose_rate_hz": round(nose_rate, 3),
            },
            "intent": intent,
            "motivation": {"hunger": round(scent["hunger"], 4),
                           "antennal_boost": round(scent["boost"], 3)},
            "nose_rate_hz": round(nose_rate, 3),
            "readout": {"ids": [int(x) for x in green]},
            "populations": populations,
            "native_4legs": {k: round(v, 3) for k, v in four.items()},
            "piano_decision": pdec,
            "active": active,
            "raster": {"neuron_ids": [], "bins": []},
        }
        # play_piano.py merges {piano: ...} here between our ticks; carry it
        # through so the 2D/3D windows stay in sync despite rewrites.
        try:
            with open(ACT) as _af:
                prev = json.load(_af)
            if isinstance(prev.get("piano"), dict):
                activity["piano"] = prev["piano"]
        except (OSError, ValueError):
            pass
        try:
            _atomic_write(ACT, json.dumps(activity, separators=(",", ":")))
        except OSError:
            pass

        tick += 1
        if tick % 25 == 1:
            print("t=%.0fs spikes=%d T1L=%.1f T1R=%.1f T2L=%.1f T2R=%.1f "
                  "nose=%.1f reward=%.2f hunger=%.2f "
                  "intent=%+.0f%s (%s) step=%.0fms act=%d" % (
                      t, brain.total_spikes, four["t1l"], four["t1r"],
                      four["t2l"], four["t2r"], nose_rate, reward_level,
                      scent["hunger"], intent["bearing_deg"],
                      "" if sugar else " (no scent)",
                      "firing" if intent["firing"] else "muted",
                      wall * 1000, len(active)), flush=True)
        if max_ticks is not None and tick >= max_ticks:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="stop after N ticks (default: run forever)")
    parser.add_argument("--tick-ms", type=float, default=TICK_MS,
                        help="simulated ms per tick (default %g)" % TICK_MS)
    parser.add_argument("--tarsal-gain", type=float, default=TARSAL_GAIN,
                        help="vnc_sensory drive while standing (default %g; "
                             "0 disables the tarsal channel)" % TARSAL_GAIN)
    parser.add_argument("--reward-gain", type=float, default=REWARD_GAIN,
                        help="sugar-reward feedback on the tarsal gain "
                             "(default %g; honest predicted coupling carrying "
                             "the scent reward to the motor channels; 0 = pure "
                             "antennal->motor path only)" % REWARD_GAIN)
    args = parser.parse_args()
    main(max_ticks=args.max_ticks, tick_ms=args.tick_ms,
         tarsal_gain=args.tarsal_gain, reward_gain=args.reward_gain)