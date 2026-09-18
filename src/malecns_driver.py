"""MaleCNS -> 3D-body driver. The 166,700-neuron male brain pilots the fly.

Loop (doomfly-env, ~3-4 ticks/s, weights frozen, no Doom, no training):
  procedural optic-flow field -> 3,335 retinal inputs
  + random sugar pulses (reward stimulus, seeded)
  -> NativeBrain.step -> bci decode (turn DNp20 R-L, forward DNpe017)
  -> walkL/walkR/startle RAW values -> $FLY_CMD  (dance_cmd.json)
  -> live neuron activity snapshot -> $FLY_ACTIVITY  (malecns_activity.json)
  -> mirror.py walks the MuJoCo body, brain_view.py draws the firing.

Mapping is an engineered readout (documented, not biology):
  total01   = forward / 20          (DNpe017 summed rate, clamped 0..20)
  diff01    = turn / 6              (DNp20 right-minus-left, clamped +-6)
  walkL/R   = 100 * clamp(total01 +/- diff01 * 0.5)
  attack    = DNpe017 spike this tick -> startle behavior + startle value
  behavior  = startle | walk | feed (during sugar, calm) | idle
No flight command exists in this decoder: the body never takes off here.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

# One BLAS thread. The per-tick readout dots are tiny, but OpenBLAS spin-waits
# its whole thread pool after every call: the driver showed 16 threads / ~500%
# CPU (~4 cores burned doing nothing) and froze screen recording. Set BEFORE
# numpy is imported; one thread is instant at these sizes.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path("/home/noirinfini/doomfly")
sys.path.insert(0, str(ROOT))
from native_brain import NativeBrain  # doom-free, ours (§NativeBrain)

# IPC paths — fly.sh exports FLY_CMD / FLY_ACTIVITY into $XDG_RUNTIME_DIR/fly/
# (user-private, mode 700). Fallback to /tmp so the script still works when
# invoked manually without fly.sh (e.g. --max-ticks smoke test).
CMD = os.environ.get("FLY_CMD",      "/tmp/dance_cmd.json")
ACT = os.environ.get("FLY_ACTIVITY", "/tmp/malecns_activity.json")
READOUT = "/home/noirinfini/fly/data/leg_readout.npz"
RECOG = "/home/noirinfini/fly/data/recognize.npz"
PIANO = "/home/noirinfini/fly/data/piano_v2.npz"
FEATHER = "/home/noirinfini/fly/data/mcns_annotations.feather"
# Visual-position stimuli (must match train_recognize.py POSITIONS).
RECOG_POS = {"left": (0.15, 0.5), "right": (0.85, 0.5),
             "top": (0.5, 0.15), "bottom": (0.5, 0.85)}
# Piano classes -> MIDI note names (pentatonic: C D E G A, + octave C6).
# C6 is the only one on the keybed's positive-y (T1_left) half — the C4..C5
# block is all negative-y (T1_right), per key_positions.json.
PIANO_KEY_NOTE = {"key0_C": "C4", "key1_D": "D4", "key2_E": "E4",
                  "key3_G": "G4", "key4_A": "A4", "key5_C2": "C6"}
# Intended-note loop (pentatonic, up and back down) — the "score" the brain
# is asked to play. Blob is presented at each key's retinal position.
PIANO_SONG = ["key0_C", "key1_D", "key2_E", "key3_G", "key4_A",
              "key5_C2", "key4_A", "key3_G", "key2_E", "key1_D"]
# DN left/right prior blend strength (log-domain). A/B (2026-09-18, task2.md):
# alpha=0.25 gave no measurable benefit (0.867 vs 0.882, n=15/17) — LEFT AT 0
# (no prior) by default; the mechanism stays behind --piano-alpha.
DN_ALPHA = 0.0
DN_PRIOR_BETA = 1.6        # prior sharpness (logit slope across the 6 keys).
# Reward-gated RPE decoder learning (piano mode). Reward = decision-match only:
# the fly earns when its decoded note == the intended note. Update once per
# decision window with a UNIT-norm feature (raw z has ||z||~sqrt(d)~116, an
# unnormalised step would wreck the ridge in one go). Gate (--rpe-gate):
#   all      update every window       (only safe at lr <= ~0.003)
#   errors   update only on misses     (default; inert when always right)
#   lowconf  update when p(intent) < --rpe-conf
# Cold = compute but never apply (log for offline replay); warm = apply live
# + per-epoch recovery toward the saved piano_v2 weights.
RPE_LR = 0.03
RPE_GATE = "errors"
RPE_CONF = 0.8
RPE_INIT_DROP = 0.0  # fraction of decoder readout rows zeroed at start: models
RPE_INIT_SEED = 7    # a "not fully wired" fly so there are mistakes to re-learn
                     # (multiplicative noise is swamped: the decoder's margins
                     # are large, so dropout is the honest imperfect init)
RPE_EPOCH = len(PIANO_SONG)   # windows per epoch = one full song pass
RPE_RECOVER_BETA = 0.5        # pull-back fraction on a worse epoch
RPE_COLD_DUMP = "/home/noirinfini/fly/data/rpe_cold.npz"
RPE_WARM_LOG = "/tmp/rpe_warm.tsv"
CUE_BLOCK = (12, 20)          # ticks per presented cue (~3-5 s at ~4 ticks/s)
LEG_L = {"left": 1.0, "right": 0.0, "top": 1.0, "bottom": 0.0}
LEG_R = {"left": 0.0, "right": 1.0, "top": 1.0, "bottom": 0.0}
CUE_SEED = 2024
# Native 4-leg mode: tarsal sensory gain (g=50 → ~55 Hz full vnc_motor pool).
NATIVE4_GAIN = 50.0
# Reward system: mean pool rate at which reward reaches maximum.
# Below this, reward scales linearly with movement; above, it saturates.
NATIVE4_REWARD_MAX_HZ = 35.0
# Sugar drive range: base (silent pools) to max (full activation).
NATIVE4_SUGAR_BASE = 30.0
NATIVE4_SUGAR_MAX  = 55.0
# Tarsal sensory gain range: base to max (scales with movement).
NATIVE4_TARSAL_BASE = 30.0
NATIVE4_TARSAL_MAX  = 80.0
TICK_MS = 28.6
# Random sugar schedule (seeded) instead of a fixed period: on SUGAR_ON_S, off
# SUGAR_OFF_S. Seeded so a run is reproducible; vary SUGAR_SEED for a new stream.
SUGAR_ON_S = (1.5, 3.0)
SUGAR_OFF_S = (6.0, 15.0)
SUGAR_SEED = 1234

# Fixed raster sample: same idea as server.py's display set, built straight
# from the graph's superclass labels (no pandas / annotations needed).
DISPLAY_GROUPS = ("ol_intrinsic", "visual_projection", "cb_intrinsic", "descending_neuron")
DISPLAY_PER_GROUP = 32
RASTER_BINS = 60
TOP_K = 800  # most-active neurons per tick, for the live 3D view


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _build_display(superclass):
    """Ascending graph indices sampled evenly across the visual/central/descending groups."""
    display = []
    for group in DISPLAY_GROUPS:
        inds = np.flatnonzero(superclass == group)
        if len(inds) == 0:
            continue
        take = min(DISPLAY_PER_GROUP, len(inds))
        display.extend(inds[np.linspace(0, len(inds) - 1, take, dtype=int)].tolist())
    return np.asarray(display, dtype=np.int64)


def _lag_feature(buf, lags):
    """[rates(t), rates(t-1), ..., rates(t-lags)] with zeros until warm."""
    arr = list(buf)  # oldest .. newest
    if len(arr) < lags + 1:
        arr = [np.zeros_like(arr[0])] * (lags + 1 - len(arr)) + arr if arr else []
    if not arr:
        return None
    return np.concatenate(arr[::-1]).astype(np.float32)


def _load_readout(path):
    """Ridge readout trained by train_leg_readout.py; None if unavailable.

    NOTE (deviation from fly.md S9): the trained target is the SMOOTHED sugar
    envelope and the decoder uses `lags` ticks of history, so the live command
    is a linear ridge output clipped to 0..1 - NOT a sigmoid of an instantaneous
    rate (a sigmoid over a 0..1 target would collapse to ~0.5-0.73). The driver
    must feed the same lagged feature vector the trainer did.
    """
    try:
        d = np.load(path)
    except (OSError, ValueError) as e:
        print(f"leg readout unavailable ({e}); leg_cmd stays 0.0", flush=True)
        return None
    r = {k: d[k] for k in ("idx", "mu", "sd", "w", "b", "lags", "topk", "topk_ids",
                           "p_lo", "p_hi")}
    r["lags"] = int(r["lags"])
    # Per-cell importance = max |weight| over its lag columns (topk was chosen so).
    r["importance"] = np.abs(r["w"].reshape(r["lags"] + 1, len(r["idx"]))).max(axis=0)
    r["lo"], r["hi"] = float(r["p_lo"]), float(r["p_hi"])
    print(f"leg readout: {len(r['idx'])} cells, lags={r['lags']}, "
          f"w={r['w'].shape[0]}, cmd range [{r['lo']:.3f},{r['hi']:.3f}]", flush=True)
    return r


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def _load_recognition(path):
    """Multiclass visual-position classifier from train_recognize.py."""
    try:
        d = np.load(path, allow_pickle=True)
    except (OSError, ValueError) as e:
        print(f"recognition model unavailable ({e}); recognition off", flush=True)
        return None
    r = {k: d[k] for k in ("idx", "mu", "sd", "W", "b", "lags", "classes", "topk_ids")}
    r["lags"] = int(r["lags"])
    r["classes"] = [str(c) for c in r["classes"]]
    r["legL"] = np.array([LEG_L.get(c, 0.0) for c in r["classes"]], np.float32)
    r["legR"] = np.array([LEG_R.get(c, 0.0) for c in r["classes"]], np.float32)
    # Piano model saves a per-class retinal position; recognition falls back
    # to RECOG_POS below. Key: class name -> (cx, cy) blob centre.
    r["pos"] = None
    if "positions" in d:
        pos = {}
        for name, cx, cy in d["positions"]:
            pos[str(name)] = (float(cx), float(cy))
        r["pos"] = pos
    print(f"recognition: {len(r['idx'])} cells, lags={r['lags']}, "
          f"classes={r['classes']}", flush=True)
    return r



def _load_native4_pools(brain):
    """Partition vnc_motor neurons by segment+side using mcns_annotations.feather.

    Returns dict with graph-index arrays for T1L/T1R/T2L/T2R motor pools and
    the vnc_sensory graph indices needed for tarsal extra_drive un-gating.
    """
    df = pd.read_feather(FEATHER)
    id_to_idx = {int(bid): i for i, bid in enumerate(brain.ids)}

    def _pool(neuromere, side):
        mask = ((df["superclass"] == "vnc_motor") &
                (df["somaNeuromere"] == neuromere) &
                (df["somaSide"] == side))
        bids = df.loc[mask, "bodyId"].astype(int).values
        idxs = np.array([id_to_idx[b] for b in bids if b in id_to_idx], dtype=np.int64)
        return idxs

    t1l = _pool("T1", "L")
    t1r = _pool("T1", "R")
    t2l = _pool("T2", "L")
    t2r = _pool("T2", "R")
    vnc_s = np.flatnonzero(brain.superclass == "vnc_sensory")  # tarsal un-gating

    print(f"native4 pools: T1L={len(t1l)} T1R={len(t1r)} "
          f"T2L={len(t2l)} T2R={len(t2r)} vnc_sensory={len(vnc_s)}", flush=True)
    return {"t1l": t1l, "t1r": t1r, "t2l": t2l, "t2r": t2r, "vnc_s": vnc_s}


def main(max_ticks=None, native4=False, piano=False, piano_alpha=DN_ALPHA,
         rpe_mode="off", rpe_lr=RPE_LR, rpe_gate=RPE_GATE,
         rpe_conf=RPE_CONF, rpe_init_drop=RPE_INIT_DROP,
         cue_seed=CUE_SEED):
    manifest = json.loads((ROOT / "outputs/doom/malecns_v1/manifest.json").read_text())
    brain = NativeBrain(ROOT / "outputs/doom/malecns_v1/graph.npz")
    controls = NeuralControls(manifest["readouts"], mode="bci")
    readout = _load_readout(READOUT)
    leg_buf = deque(maxlen=(readout["lags"] + 1) if readout else 1)
    # Piano mode: decode the 6-key piano model instead of the 4-cue one.
    recog = _load_recognition(PIANO if piano else RECOG)
    recog_buf = deque(maxlen=(recog["lags"] + 1) if recog else 1)
    cue_rng = np.random.default_rng(cue_seed)
    cue_idx, cue_until = 0, -1
    if recog is not None:
        recog_pos = RECOG_POS if recog["pos"] is None else recog["pos"]
        lat = np.linspace(0.0, 1.0, len(recog["classes"]))  # 0=left key..1=right
    p_smooth = np.zeros(len(recog["classes"]), np.float64) if recog else None
    song_i = 0
    p_ok_n = p_tot_n = 0
    pdec = {"decision": "none", "intent": "none", "ok": False, "conf": 0.0,
            "scores": [], "bias": 0.0, "alpha": piano_alpha, "song_i": -1,
            "n_ok": 0, "n_tot": 0, "motor": {"T1L": 0.0, "T1R": 0.0}}
    # --- Reward-gated RPE learning state (piano mode only) ---
    rpe = {"mode": rpe_mode, "lr": rpe_lr, "gate": rpe_gate, "conf": rpe_conf,
           "win": 0, "ep_win": 0, "ep_ok": 0,
           "epoch": 0, "acc": 0.0, "best": 0.0, "recovered": False,
           "updates": 0, "z": None, "p": None, "choice": -1,
           "zlog": [], "plog": [], "ilog": [], "clog": []}
    if piano and recog is not None:
        recog["W"] = recog["W"].astype(np.float64).copy()
        recog["b"] = recog["b"].astype(np.float64).copy()
        rpe["W0"] = recog["W"].copy()
        rpe["b0"] = recog["b"].copy()
        if rpe_init_drop > 0:
            dz = np.random.default_rng(RPE_INIT_SEED)
            keep = dz.random(recog["W"].shape[0]) > rpe_init_drop
            recog["W"] *= keep[:, None]
        if rpe_mode == "warm":
            try:
                open(RPE_WARM_LOG, "w").close()
            except OSError:
                pass

    def rpe_recover():
        """Pull the working decoder halfway back toward piano_v2's weights."""
        w0 = np.asarray(rpe["W0"]); b0 = np.asarray(rpe["b0"])
        recog["W"] = w0 + RPE_RECOVER_BETA * (recog["W"] - w0)
        recog["b"] = b0 + RPE_RECOVER_BETA * (recog["b"] - b0)

    def rpe_window_end(ended_song):
        """One decision window just ended: score it by its LOCKED decision (the
        last posterior of the window — the choice play_piano would press), then
        reward-gated update + epoch books if warm/cold."""
        nonlocal p_ok_n, p_tot_n
        if rpe["z"] is None:
            return
        intent_idx = recog["classes"].index(PIANO_SONG[ended_song])
        pv = np.asarray(rpe["p"])
        choice = int(rpe["choice"])
        ok = choice == intent_idx
        rpe["win"] += 1
        rpe["ep_win"] += 1
        rpe["ep_ok"] += int(ok)
        p_tot_n += 1
        p_ok_n += int(ok)
        if rpe["mode"] == "cold":
            rpe["zlog"].append(rpe["z"].astype(np.float32))
            rpe["plog"].append(pv.astype(np.float32))
            rpe["ilog"].append(intent_idx)
            rpe["clog"].append(choice)
        if rpe["mode"] == "warm":
            g = rpe["gate"]
            do_update = (g == "all") or (g == "errors" and not ok) or (
                g == "lowconf" and float(pv[intent_idx]) < rpe["conf"])
            if do_update:
                y = np.zeros(len(recog["classes"]), np.float64)
                y[intent_idx] = 1.0
                zn = np.asarray(rpe["z"]) / (float(np.linalg.norm(rpe["z"])) + 1e-8)
                recog["W"] += rpe["lr"] * np.outer(zn, y - pv)
                recog["b"] += rpe["lr"] * (y - pv)
                rpe["updates"] += 1
            if rpe["ep_win"] >= RPE_EPOCH:
                acc = rpe["ep_ok"] / rpe["ep_win"]
                rpe["epoch"] += 1
                rpe["acc"] = acc
                if acc >= rpe["best"]:
                    rpe["best"] = acc
                    rpe["recovered"] = False
                else:
                    rpe_recover()
                    rpe["recovered"] = True
                rpe["ep_win"] = 0
                rpe["ep_ok"] = 0
            try:
                with open(RPE_WARM_LOG, "a") as f:
                    f.write("%d\t%d\t%d\t%d\t%.4f\t%d\t%d\t%d\n" % (
                        rpe["win"], intent_idx, choice, int(ok),
                        float(pv[intent_idx]), rpe["epoch"],
                        int(rpe["recovered"]), rpe["updates"]))
            except OSError:
                pass
        rpe["z"] = None  # consumed; next tick re-captures for the new window

    def rpe_dump_cold():
        if rpe["mode"] == "cold" and rpe["zlog"]:
            np.savez(RPE_COLD_DUMP, z=np.array(rpe["zlog"]),
                     p=np.array(rpe["plog"]), i=np.array(rpe["ilog"]),
                     c=np.array(rpe["clog"]), lr=np.float32(rpe["lr"]),
                     seed=np.int32(cue_seed))
            print(f"rpe cold dump -> {RPE_COLD_DUMP} "
                  f"({len(rpe['zlog'])} windows)", flush=True)

    uv = brain.uv  # (3335, 2) receptor coordinates
    n = len(brain.retina)
    group_names = np.unique(brain.superclass)
    groups = [np.flatnonzero(brain.superclass == k) for k in group_names]
    display = _build_display(brain.superclass)
    display_ids = [str(x) for x in brain.ids[display]]
    history = deque(maxlen=RASTER_BINS)
    sugar_rng = np.random.default_rng(SUGAR_SEED)
    sugar_until = 0.0
    sugar_next = float(sugar_rng.uniform(*SUGAR_OFF_S))
    # Native pools: full 4-leg drive in --native-4legs, and T1L/T1R motor rates
    # for the piano press envelope in --piano.
    n4pools = _load_native4_pools(brain) if (native4 or piano) else None
    # prev_n4_rates: rates from the PREVIOUS tick used to compute THIS tick's reward.
    # Starts at zero (no movement yet → base drives only).
    prev_n4_rates = {"T1L": 0.0, "T1R": 0.0, "T2L": 0.0, "T2R": 0.0}
    # Pre-build the sugar cell index list (for proportional sugar drive).
    # brain.sugar is a direct attribute — index array of LB3c sugar cells.
    n4_sugar_idxs = brain.sugar.tolist() if (native4 and hasattr(brain, 'sugar') and brain.sugar is not None) else []
    print(f"malecns driver: {brain.n} neurons, {n} retinal inputs, "
          f"{len(display)} display neurons, {len(group_names)} populations"
          f"{' [native4]' if native4 else ''}", flush=True)
    t0 = time.monotonic()
    tick = 0
    while True:
        t = time.monotonic() - t0
        # --- present a visual cue: random 4-cue (recognition) OR the intended
        # piano note's key position (piano mode) ---
        if recog is not None and tick >= cue_until:
            if piano:
                rpe_window_end(song_i)      # ended window's decision -> RPE
                song_i = (song_i + 1) % len(PIANO_SONG)
                cue_idx = recog["classes"].index(PIANO_SONG[song_i])
            else:
                cue_idx = int(cue_rng.integers(0, len(recog["classes"])))
            cue_until = tick + int(cue_rng.integers(*CUE_BLOCK))
        cue = recog["classes"][cue_idx] if recog is not None else None
        if recog is not None:
            # Match train_recognize.py: dim deterministic optic flow + a blob.
            tv = tick * TICK_MS / 1000.0
            fd = 1.0 if math.sin(tv * 0.11) >= 0 else -1.0
            phase = uv[:, 0] * 12.0 - tv * 3.0 * fd + uv[:, 1] * 4.0
            lum = 0.1 * np.where(np.sin(2 * math.pi * phase) > 0, 1.0, 0.05).astype(np.float32)
            cx, cy = recog_pos[cue]
            d2 = (uv[:, 0] - cx) ** 2 + (uv[:, 1] - cy) ** 2
            lum = np.clip(lum + np.exp(-d2 / (2 * 0.18 ** 2)).astype(np.float32), 0, 1)
        else:
            # Procedural visual world: high-contrast drifting bars = optic flow.
            flow_dir = 1.0 if math.sin(t * 0.11) >= 0 else -1.0
            phase = uv[:, 0] * 12.0 - t * 3.0 * flow_dir + uv[:, 1] * 4.0
            lum = np.where(np.sin(2 * math.pi * phase) > 0, 1.0, 0.05).astype(np.float32)
            if (t % 7.0) < 0.15:
                lum[:] = 1.0
            lum += np.random.default_rng(tick).normal(0, 0.03, n).astype(np.float32)
            lum = np.clip(lum, 0, 1)
        # Random sugar schedule (seeded): on SUGAR_ON_S, off SUGAR_OFF_S.
        if t >= sugar_next:
            sugar_until = t + float(sugar_rng.uniform(*SUGAR_ON_S))
            sugar_next = sugar_until + float(sugar_rng.uniform(*SUGAR_OFF_S))
        sugar = t < sugar_until
        # In native4 mode: proportional sugar + proportional tarsal sensory un-gating.
        # Reward scales with PREVIOUS tick's mean motor pool rate (closed-loop feedback).
        if native4 and n4pools is not None:
            mean_prev = sum(prev_n4_rates.values()) / 4.0
            movement_frac = float(np.clip(mean_prev / NATIVE4_REWARD_MAX_HZ, 0.0, 1.0))
            sugar_drive = NATIVE4_SUGAR_BASE + (NATIVE4_SUGAR_MAX - NATIVE4_SUGAR_BASE) * movement_frac
            tarsal_gain = NATIVE4_TARSAL_BASE + (NATIVE4_TARSAL_MAX - NATIVE4_TARSAL_BASE) * movement_frac
            n4_extra = [(int(i), tarsal_gain) for i in n4pools["vnc_s"]]
            # Inject proportional sugar drive into sugar cells (overrides sugar=False).
            if n4_sugar_idxs:
                n4_extra += [(int(i), sugar_drive) for i in n4_sugar_idxs]
                counts, wall = brain.step(lum, TICK_MS, sugar=False, extra_drive=n4_extra)
            else:
                # Fallback: use sugar=True boolean + tarsal drive only.
                counts, wall = brain.step(lum, TICK_MS, sugar=True, extra_drive=n4_extra)
        else:
            movement_frac = 0.0
            counts, wall = brain.step(lum, TICK_MS, sugar=bool(sugar))
        a = controls.decode(counts, TICK_MS / 1000)
        tick_s = TICK_MS / 1000.0
        # Trained reservoir readout: smoothed-sugar -> foreleg-lift command.
        leg_cmd = 0.0
        if readout is not None:
            leg_buf.append(counts[readout["idx"]] / tick_s)
            feat = _lag_feature(leg_buf, readout["lags"])
            if feat is not None:
                z = (feat - readout["mu"]) / readout["sd"]
                pred = float(readout["w"] @ z + readout["b"])
                leg_cmd = float(np.clip((pred - readout["lo"]) /
                                        (readout["hi"] - readout["lo"]), 0.0, 1.0))
        # Multiclass recognition of the presented visual cue (optic-lobe cells).
        legL_cmd = legR_cmd = 0.0
        rec_class, rec_conf, rec_ok, rec_scores = "none", 0.0, False, []
        dn_bias = 0.0
        if recog is not None:
            recog_buf.append(counts[recog["idx"]] / tick_s)
            feat_r = _lag_feature(recog_buf, recog["lags"])
            if feat_r is not None:
                zr = (feat_r - recog["mu"]) / recog["sd"]
                p = _softmax(recog["W"].T @ zr + recog["b"])
                p_smooth = 0.6 * p_smooth + 0.4 * p
                if piano:
                    # Native DN left/right bias -> per-key prior. turn>0 means
                    # DNp20 right-minus-left positive (leans right); leftness<0.
                    leftness = -_clamp(a["turn"] / 6.0, -1.0, 1.0)  # >0 = leans left
                    anc = (1.0 - 2.0 * lat)                          # +1 at leftmost key
                    log_prior = DN_PRIOR_BETA * leftness * anc
                    if piano_alpha > 0:
                        p_blend = p_smooth * np.exp(piano_alpha * log_prior)
                        p_blend /= p_blend.sum()
                    else:
                        p_blend = p_smooth
                    dn_bias = float(leftness)
                    ci = int(np.argmax(p_blend))
                    chosen = recog["classes"][ci]
                    # RPE capture: the deciding posterior + the feature vector
                    # that produced it (for the window that ends at the next
                    # boundary). Unit-norm on the z before any update.
                    rpe["z"] = zr.copy()
                    rpe["p"] = p_blend.copy()
                    rpe["choice"] = ci
                    pdec["decision"] = chosen
                    pdec["intent"] = cue
                    pdec["ok"] = chosen == cue
                    pdec["conf"] = float(p_blend[ci])
                    pdec["scores"] = [round(float(x), 3) for x in p_blend]
                    pdec["bias"] = round(dn_bias, 3)
                    pdec["alpha"] = piano_alpha
                    pdec["song_i"] = song_i
                    pdec["n_ok"] = p_ok_n
                    pdec["n_tot"] = p_tot_n
                else:
                    ci = int(np.argmax(p_smooth))
                rec_class = recog["classes"][ci]
                rec_conf = float(p_smooth[ci])
                rec_ok = bool(rec_class == cue)
                rec_scores = [round(float(x), 3) for x in p_smooth]
                # Commands use the (smoothed) argmax class one-hot: the posterior
                # is deliberately diffuse offline, but the winner is stable, and
                # the body's first-order filter supplies the ramp.
                if not piano:
                    legL_cmd = float(recog["legL"][ci])
                    legR_cmd = float(recog["legR"][ci])
        # Forward from summed DNp20 rate (same two steering cells): DNpe017
        # either flatlines (smooth scenes) or clamps at 20 (hard bars), so it
        # carries no proportional signal. DNp20 sum varies 3-26Hz = real gas.
        dn_sum = sum(r["rate_hz"] for r in a["readouts"] if r["type"] == "DNp20")
        total01 = _clamp(dn_sum / 30.0, 0.0, 1.0)
        diff01 = _clamp(a["turn"] / 6.0, -1.0, 1.0)
        walkL = 100.0 * _clamp(total01 + diff01 * 0.5, 0.0, 1.0)
        walkR = 100.0 * _clamp(total01 - diff01 * 0.5, 0.0, 1.0)
        if a["attack"]:
            behavior, startle = "startle", 100.0
        elif total01 > 0.15:
            behavior, startle = "walk", 0.0
        elif sugar:
            behavior, startle = "feed", 0.0
        else:
            behavior, startle = "idle", 0.0
        out = {"walkL": round(walkL, 2), "walkR": round(walkR, 2),
               "startle": startle, "flight": 0.0, "groom": 0.0,
               "feed": 100.0 if sugar else 0.0, "behavior": behavior,
               "fear": 0.0, "hunger": 0.0, "curiosity": 0.5,
               "frozen": False, "brain": "malecns",
               "turn": round(a["turn"], 2), "fwd": round(a["forward"], 2)}
        try:
            _atomic_write(CMD, json.dumps(out))
        except OSError:
            pass

        # Live neuron activity snapshot for brain_view.py (same data the Doom
        # backend used to serve, now written locally with no HTTP).
        history.append([int(x) for x in counts[display]])
        # Top-K most active neurons this tick (body id + spike count) so the
        # 3D view can light up far more than the 128 raster channels.
        nz = np.flatnonzero(counts)
        if len(nz) > TOP_K:
            nz = nz[np.argpartition(counts[nz], -TOP_K)[-TOP_K:]]
        active = [{"id": int(brain.ids[i]), "n": int(counts[i])} for i in nz]
        populations = [{"name": str(k), "neurons": int(len(ix)),
                        "spikes": int(counts[ix].sum()),
                        "mean_rate_hz": round(float(counts[ix].sum() / max(len(ix), 1) / tick_s), 3)}
                       for k, ix in zip(group_names, groups)]
        readouts = [{"id": str(r["id"]), "type": r["type"], "side": r["side"],
                     "spikes": int(r["spikes"]), "rate_hz": round(float(r["rate_hz"]), 3),
                     "voltage_mv": round(float(brain.v[r["index"]]), 3)}
                    for r in a["readouts"]]
        # Native 4-leg pool rates (only in --native-4legs mode).
        native_4legs = None
        if n4pools is not None:
            tick_s_n4 = TICK_MS / 1000.0
            def _pool_rate(idxs):
                return round(float(counts[idxs].sum() / max(len(idxs), 1) / tick_s_n4), 3)
            native_4legs = {
                "T1L": _pool_rate(n4pools["t1l"]),
                "T1R": _pool_rate(n4pools["t1r"]),
                "T2L": _pool_rate(n4pools["t2l"]),
                "T2R": _pool_rate(n4pools["t2r"]),
            }
            # Store rates for NEXT tick's reward computation.
            prev_n4_rates = dict(native_4legs)
            if piano:
                pdec["motor"] = {"T1L": native_4legs["T1L"],
                                 "T1R": native_4legs["T1R"]}
        activity = {"status": "running", "tick": tick, "sequence": tick,
                    "generated_at_ms": int(time.time() * 1000),
                    "wall_s": round(t, 2), "step_ms": round(wall * 1000, 1),
                    "sim_ms": round(brain.sim_ms, 1), "total_spikes": brain.total_spikes,
                    "tick_spikes": int(counts.sum()),
                    "behavior": behavior, "sugar": bool(sugar),
                    "leg_cmd": round(leg_cmd, 4), "leg_sugar": bool(sugar),
                    "cue": cue,
                    "recog": {"class": rec_class, "conf": round(rec_conf, 3),
                               "ok": rec_ok, "scores": rec_scores},
                    "piano_decision": pdec,
                    "dn_bias": round(dn_bias, 3),
                    "piano_learning": {"mode": rpe["mode"], "lr": rpe["lr"],
                                       "gate": rpe_gate,
                                       "init_drop": rpe_init_drop,
                                       "window": rpe["win"],
                                       "epoch": rpe["epoch"],
                                       "acc_epoch": round(rpe["acc"], 3),
                                       "best_acc": round(rpe["best"], 3),
                                       "recovered": rpe["recovered"],
                                       "updates": rpe["updates"]},
                    "legL_cmd": round(legL_cmd, 4), "legR_cmd": round(legR_cmd, 4),
                    "readout": ({"ids": [int(x) for x in recog["topk_ids"]], "weights": []}
                                if recog is not None
                                else ({"ids": [int(x) for x in readout["topk_ids"]],
                                       "weights": [round(float(readout["importance"][int(p)]), 5)
                                                   for p in readout["topk"]]}
                                      if readout is not None else {"ids": [], "weights": []})),
                    "action": {"turn": round(a["turn"], 3),
                               "forward": round(a["forward"], 3),
                               "attack": bool(a["attack"])},
                    "populations": populations, "readouts": readouts,
                    "active": active,
                    "raster": {"neuron_ids": display_ids, "bins": list(history)}}
        if native_4legs is not None:
            activity["native_4legs"] = native_4legs
            activity["reward_level"] = round(movement_frac, 3)
            activity["reward_frac"] = round(movement_frac, 3)
        if piano:
            # play_piano.py merges {piano: ...} into this file between
            # driver ticks; carry the latest note through so the 2D/3D
            # windows stay in sync even though we rewrite the file.
            try:
                with open(ACT) as _f:
                    _prev = json.load(_f)
                if isinstance(_prev.get("piano"), dict):
                    activity["piano"] = _prev["piano"]
            except (OSError, ValueError):
                pass
        try:
            _atomic_write(ACT, json.dumps(activity, separators=(",", ":")))
        except OSError:
            pass

        tick += 1
        if tick % 40 == 1:
            if native4 and native_4legs is not None:
                reward_bar = "█" * int(movement_frac * 8) + "░" * (8 - int(movement_frac * 8))
                print(f"t={t:.0f}s spikes={brain.total_spikes} "
                      f"T1L={native_4legs['T1L']:.1f}Hz T1R={native_4legs['T1R']:.1f}Hz "
                      f"T2L={native_4legs['T2L']:.1f}Hz T2R={native_4legs['T2R']:.1f}Hz "
                      f"reward={reward_bar}({movement_frac:.2f}) "
                      f"sugar={sugar_drive:.0f} tarsal={tarsal_gain:.0f} "
                      f"step={wall*1000:.0f}ms", flush=True)
            else:
                msg = (f"t={t:.0f}s spikes={brain.total_spikes} "
                       f"DNp20={dn_sum:.1f}Hz turn={a['turn']:+.2f} "
                       f"beh={behavior} sugar={sugar} "
                       f"cue={cue} recog={rec_class}({rec_conf:.2f})"
                       f"{'ok' if rec_ok else 'X'} step={wall*1000:.0f}ms")
                if piano:
                    msg += (f"  PIANO intent={pdec['intent']} "
                            f"decide={pdec['decision']} "
                            f"bias={pdec['bias']:+.2f} "
                            f"{'OK' if pdec['ok'] else 'miss'}")
                    if rpe["mode"] != "off":
                        msg += (f" rpe[{rpe['mode']}] e{rpe['epoch']} "
                                f"acc={rpe['acc']:.2f} best={rpe['best']:.2f} "
                                f"{'R' if rpe['recovered'] else ''}")
                print(msg, flush=True)
        if max_ticks is not None and tick >= max_ticks:
            rpe_dump_cold()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="stop after N ticks (default: run forever)")
    parser.add_argument("--native-4legs", action="store_true",
                        help="drive T1L/T1R/T2L/T2R from native vnc_motor rates "
                             "(no trained decoder; tri-stimulus: sugar+optic+tarsal)")
    parser.add_argument("--piano", action="store_true",
                        help="piano mode: present the intended note's key "
                             "position, decode the 6-key piano model with a "
                             "DN left/right prior blend, and emit the fly's "
                             "chosen note in piano_decision; play_piano.py "
                             "presses it")
    parser.add_argument("--piano-alpha", type=float, default=DN_ALPHA,
                        help="DN prior blend strength in piano mode "
                             "(0 = no prior, for A/B measurement)")
    parser.add_argument("--rpe", default="off", choices=("off", "cold", "warm"),
                        help="reward-gated RPE learning in piano mode: cold = "
                             "compute but never apply (logs for offline replay); "
                             "warm = apply live with per-epoch recovery")
    parser.add_argument("--rpe-lr", type=float, default=RPE_LR,
                        help="RPE learning rate (default %g)" % RPE_LR)
    parser.add_argument("--rpe-gate", default=RPE_GATE,
                        choices=("all", "errors", "lowconf"),
                        help="which windows may update warm-mode weights: "
                             "errors = only on misses (default); lowconf = "
                             "whenever p(intended) < --rpe-conf; all = every "
                             "window (only safe at lr <= ~0.003)")
    parser.add_argument("--rpe-conf", type=float, default=RPE_CONF,
                        help="lowconf gate threshold (default %g)" % RPE_CONF)
    parser.add_argument("--rpe-init-drop", type=float, default=RPE_INIT_DROP,
                        help="fraction of STARTING decoder readout rows zeroed "
                             "(default %g): start 'unwired' so errors exist for "
                             "error-gated RPE to fix; recovery target stays the "
                             "full saved model" % RPE_INIT_DROP)
    parser.add_argument("--cue-seed", type=int, default=CUE_SEED,
                        help="sugar/cue RNG seed (affects song timing + DN prior; "
                             "default %d)" % CUE_SEED)
    args = parser.parse_args()
    main(max_ticks=args.max_ticks, native4=args.native_4legs,
         piano=args.piano, piano_alpha=args.piano_alpha,
         rpe_mode=args.rpe, rpe_lr=args.rpe_lr,
         rpe_gate=args.rpe_gate, rpe_conf=args.rpe_conf,
         rpe_init_drop=args.rpe_init_drop,
         cue_seed=args.cue_seed)
