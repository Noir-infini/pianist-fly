"""scent_ab — headless kills: does the fly smell, want, point, and reach?

Five headless gates over the real connectome (M5/INTENT/HUNGER/ASSETS are
MuJoCo-free; REACH pulls in the piano body only for the two measured presses):

  M5       — does scent measurably raise the piano press drive? Interleaved
             ON/OFF sugar blocks at steady state; gate: scent-block mean T1
             rate >= off-block * 1.20 (delta >= +20%) and point-biserial
             corr(scent, rate) >= 0.5. Uses the uniform full-strength nose
             drive (the M5 CI), not the spatial plume.
  INTENT   — does the L/R nose point where the invisible plume is? The
             spatial source is placed RIGHT, LEFT, then AHEAD; the fly's own
             antennal L/R response (wiring-grounded split) must move away
             from its symmetric baseline toward the source, and the SHIPPED
             press readout (lateral weighting, tagged predicted) must lean
             onto the smell side while the nose is actually firing.
  HUNGER   — is appetite real effort? Same plume, hunger 0.1 vs 0.9; gate:
             at sharp hunger the nose fires (lazily hungry it stays (near)
             silent) AND T1 press rate >= lazy * 1.40 (delta >= +40%).
  ASSETS   — every note of the C-major score (C4..C6) has an audible wav.
  REACH    — sugar-reach mode split: placing the invisible sugar on an outer
             key makes the fly's OWN antennae point that way (real), and the
             leg lands exactly on the sugar key (system, measured pressed_ok).
             The key->plume mapping (keybed Y -> off-axis offset) is tagged
             `predicted` geometry.

Exit 0 = all gates pass, 2 = any fails. `--reward-gain` is the sugar-reward
coupling (tarsal_gain *= 1 + reward_gain * reward_level), an honest
brain-internal `predicted` feedback: 0 compares the pure antennal->motor path.

Run:  .venv/bin/python scripts/scent_ab.py [--blocks 6] [--block-ticks 105]
      pianist.sh verify-scent
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

for _env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env, "1")

import numpy as np

from native_brain import NativeBrain
import brain_driver as bd
import key_sound

GRAPH = HERE / "data/graph.npz"

# play_piano.py's exact force mapping (documented there; mirrored here so this
# headless test never pulls in MuJoCo/flybody).
BRAIN_FORCE_FULL_HZ = 55.0
BRAIN_FORCE_LO = 0.3
BRAIN_FORCE_HI = 1.5

# Invisible spatial source placements for the intent gate: one off RIGHT, one
# off LEFT (both off-axis so one antenna clearly out-smells the other), and
# one dead AHEAD. AHEAD doubles as the intrinsic L/R baseline: the left and
# right pools are different cells, so they do not fire equally on a symmetric
# plume — the gate compares the off-axis reads to their own baseline.
SOURCES = [("AHEAD", (0.22, 0.0)), ("RIGHT", (0.22, 0.14)),
           ("LEFT", (0.22, -0.14))]


def brain_force(rate_hz):
    if rate_hz <= 0.0:
        return 1.0
    return float(np.clip(rate_hz / BRAIN_FORCE_FULL_HZ, BRAIN_FORCE_LO,
                         BRAIN_FORCE_HI))


def _point_biserial(x, y):
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _spatial_run(brain, pools, nose_sides, vnc_sensory, source, ticks=90,
                 warmup=25, sugar_on=True, hunger_cmd=None, rng=None,
                 reward_gain=bd.REWARD_GAIN, tarsal_gain=bd.TARSAL_GAIN):
    """Run the plume through the SAME per-side drive the live driver uses.

    Returns a dict of equal-length per-measured-tick arrays."""
    model = bd.ScentModel(source)
    if rng is None:
        rng = np.random.default_rng(7)
    tick_s = bd.TICK_MS / 1000.0
    reward = 0.0
    rec = {k: [] for k in
           ("nose_l", "nose_r", "t1l", "t1r", "cL", "cR", "bearing",
            "hunger", "strength", "firing", "lat_l", "lat_r",
            "pub_t1l", "pub_t1r")}
    for tick in range(warmup + ticks):
        t = tick * tick_s
        oo = model.tick(sugar_on, source=source, hunger_cmd=hunger_cmd)
        lum = bd._optic_flow(brain.uv, t, tick, rng)
        tarsal_eff = bd.tarsal_effort(tarsal_gain, reward_gain, reward,
                                      oo["hunger"])
        extra = ([(int(i), tarsal_eff) for i in vnc_sensory]
                 if tarsal_eff > 0.0 else [])
        if sugar_on:
            extra += bd.sugar_cell_drive(oo, nose_sides)
        counts, wall = brain.step(lum, bd.TICK_MS, sugar=sugar_on,
                                  lamina_bias=bd.LAMBIA_BIAS,
                                  extra_drive=extra or None)

        def _m(idx):
            return float(counts[idx].sum() / max(len(idx), 1) / tick_s)
        nose_r, nose_l = _m(nose_sides["r"]), _m(nose_sides["l"])
        nose_rate = _m(brain.sugar)
        reward = float(np.clip(nose_rate / bd.REWARD_FULL_HZ, 0.0, 1.0))
        if tick >= warmup:
            t1l_tick, t1r_tick = _m(pools["t1l"]), _m(pools["t1r"])
            lat = bd.lateral_press(oo)
            rec["nose_r"].append(nose_r)
            rec["nose_l"].append(nose_l)
            rec["t1l"].append(t1l_tick)
            rec["t1r"].append(t1r_tick)
            rec["cL"].append(oo["cL"])
            rec["cR"].append(oo["cR"])
            rec["bearing"].append(oo["bearing_deg"])
            rec["hunger"].append(oo["hunger"])
            rec["strength"].append(0.5 * (oo["sL"] + oo["sR"]))
            rec["firing"].append(nose_rate > 0.0)
            rec["lat_l"].append(lat["T1L"])
            rec["lat_r"].append(lat["T1R"])
            rec["pub_t1l"].append(t1l_tick * lat["T1L"])
            rec["pub_t1r"].append(t1r_tick * lat["T1R"])
    return rec


def _m5_gate(brain, pools, vnc_sensory, blocks=6, block_ticks=105,
             warmup_ticks=35, tarsal_gain=bd.TARSAL_GAIN,
             reward_gain=bd.REWARD_GAIN):
    tick_s = bd.TICK_MS / 1000.0
    total = warmup_ticks + blocks * block_ticks
    sugar = np.zeros(total, dtype=bool)
    rate = np.zeros(total, dtype=np.float64)
    t1l = np.zeros(total, dtype=np.float64)
    t1r = np.zeros(total, dtype=np.float64)
    reward = 0.0
    optic_rng = np.random.default_rng(0)
    for tick in range(total):
        t = tick * tick_s
        block = max(0, (tick - warmup_ticks)) // block_ticks
        sugar[tick] = block % 2 == 1
        lum = bd._optic_flow(brain.uv, t, tick, optic_rng)
        tarsal_eff = (tarsal_gain * (1.0 + reward_gain * reward)
                      if reward_gain > 0.0 else tarsal_gain)
        extra = ([(int(i), tarsal_eff) for i in vnc_sensory]
                 if tarsal_eff > 0.0 else None)
        counts, wall = brain.step(lum, bd.TICK_MS, sugar=bool(sugar[tick]),
                                  lamina_bias=bd.LAMBIA_BIAS,
                                  extra_drive=extra)

        def _mean(idx):
            return float(counts[idx].sum() / max(len(idx), 1) / tick_s)
        t1l[tick], t1r[tick] = _mean(pools["t1l"]), _mean(pools["t1r"])
        rate[tick] = 0.5 * (t1l[tick] + t1r[tick])
        reward = float(np.clip(_mean(brain.sugar) / bd.REWARD_FULL_HZ, 0.0, 1.0))

    s = slice(warmup_ticks, total)
    on, off = rate[s][sugar[s]], rate[s][~sugar[s]]
    mu_on, mu_off = float(on.mean()), float(off.mean())
    delta = (mu_on - mu_off) / mu_off * 100.0 if mu_off > 0 else 0.0
    corr = _point_biserial(sugar[s].astype(np.float64), rate[s])
    print(f"  T1 avg:  scent ON {mu_on:5.1f} Hz vs OFF {mu_off:5.1f} Hz "
          f"-> delta +{delta:.1f}%")
    print(f"  point-biserial corr(scent, rate) = {corr:+.3f} (gate >= 0.5)")
    ok = delta >= 20.0 and corr >= 0.5
    print(f"  M5 gate: {'PASS' if ok else 'FAIL'} (delta +{delta:.1f}% >= +20% "
          f"and corr {corr:+.3f} >= 0.5)")
    return ok


def _intent_gate(brain, pools, nose_sides, vnc_sensory):
    runs = {}
    print("  intent (spatial plume through the L/R antenna):")
    for name, src in SOURCES:
        rec = _spatial_run(brain, pools, nose_sides, vnc_sensory, src,
                           hunger_cmd=0.9, rng=np.random.default_rng(3))
        runs[name] = rec
        nd = float(np.mean(rec["nose_r"]) - np.mean(rec["nose_l"]))
        pd = float(np.mean(rec["pub_t1r"]) - np.mean(rec["pub_t1l"]))
        fired = all(rec["firing"])
        true_b = math.degrees(math.atan2(src[1], src[0]))
        pub_b = float(np.mean(rec["bearing"]))
        ml, mr = float(np.mean(rec["lat_l"])), float(np.mean(rec["lat_r"]))
        print(f"    [{name:5s}] src=({src[0]:.2f},{src[1]:.2f}) "
              f"bearing_pub={pub_b:+5.1f} (true {true_b:+5.1f}) "
              f"cL={np.mean(rec['cL']):.3f} cR={np.mean(rec['cR']):.3f} "
              f"latL={ml:.3f} latR={mr:.3f} noseDel={nd:+.2f} "
              f"pubMotorDel={pd:+.2f} fired={fired}")
    base_d = float(np.mean(runs["AHEAD"]["nose_r"])
                   - np.mean(runs["AHEAD"]["nose_l"]))
    right_d = float(np.mean(runs["RIGHT"]["nose_r"])
                    - np.mean(runs["RIGHT"]["nose_l"]))
    left_d = float(np.mean(runs["LEFT"]["nose_r"])
                   - np.mean(runs["LEFT"]["nose_l"]))
    spread = max(right_d, left_d, base_d) - min(right_d, left_d, base_d)
    # The connectome's OWN nose response must move away from its symmetric
    # baseline: RIGHT raises the right antenna, LEFT raises the left one.
    side_ok = (right_d > base_d) and (left_d < base_d) and spread > 1.0
    fired_ok = all(all(r["firing"]) for r in runs.values())
    # The SHIPPED press readout (lateral weighting, tagged predicted) must
    # lean toward the smell: RIGHT plume -> T1R weighted up, LEFT -> T1L up,
    # AHEAD -> neutral.
    lat_ok = (np.mean(runs["RIGHT"]["lat_r"]) > np.mean(runs["RIGHT"]["lat_l"])
              and np.mean(runs["LEFT"]["lat_l"])
              > np.mean(runs["LEFT"]["lat_r"])
              and abs(np.mean(runs["AHEAD"]["lat_r"])
                      - np.mean(runs["AHEAD"]["lat_l"])) < 1e-6)
    pub_base = float(np.mean(runs["AHEAD"]["pub_t1r"])
                     - np.mean(runs["AHEAD"]["pub_t1l"]))
    pub_r = float(np.mean(runs["RIGHT"]["pub_t1r"])
                  - np.mean(runs["RIGHT"]["pub_t1l"]))
    pub_l = float(np.mean(runs["LEFT"]["pub_t1r"])
                  - np.mean(runs["LEFT"]["pub_t1l"]))
    pub_ok = pub_r > pub_base and pub_l < pub_base
    print(f"  nose sidedness vs AHEAD baseline {base_d:+.2f}: "
          f"RIGHT {right_d:+.2f} (>base,ok={right_d > base_d}), "
          f"LEFT {left_d:+.2f} (<base,ok={left_d < base_d}), "
          f"spread={spread:.2f}Hz (ok={spread > 1.0}) | nose firing = {fired_ok}")
    print(f"  shipped press laterality {lat_ok} | published motor sidedness "
          f"{pub_ok} (RIGHT {pub_r:+.2f} > base {pub_base:+.2f}, "
          f"LEFT {pub_l:+.2f} < base)")
    ok = side_ok and fired_ok and lat_ok and pub_ok
    print(f"  INTENT gate: {'PASS' if ok else 'FAIL'}")
    return ok


def _hunger_gate(brain, pools, nose_sides, vnc_sensory):
    print("  hunger (same plume, appetite 0.1 vs 0.9):")
    src = SOURCES[0][1]
    lazy = _spatial_run(brain, pools, nose_sides, vnc_sensory, src,
                        hunger_cmd=0.1, rng=np.random.default_rng(11))
    sharp = _spatial_run(brain, pools, nose_sides, vnc_sensory, src,
                         hunger_cmd=0.9, rng=np.random.default_rng(11))
    def _energy(r):
        return (0.5 * (np.mean(r["nose_l"]) + np.mean(r["nose_r"])),
                0.5 * (np.mean(r["t1l"]) + np.mean(r["t1r"])),
                float(np.mean(r["strength"])))
    ln, lt, ls = _energy(lazy)
    sn, st, ss = _energy(sharp)
    dn = (sn - ln) / ln * 100.0 if ln >= 1.0 else None
    dt = (st - lt) / lt * 100.0 if lt > 0 else 0.0
    print(f"    lazy : nose {ln:5.1f} Hz  T1 {lt:5.1f} Hz  intent {ls:.3f}")
    print(f"    sharp: nose {sn:5.1f} Hz  T1 {st:5.1f} Hz  intent {ss:.3f}")
    # Lazily hungry, the invisible plume barely registers (nose silent);
    # sharply hungry, it fires hard. If a future tuning ever makes both noses
    # nonzero, the quantitative delta applies instead.
    if ln < 2.0:
        nose_ok = sn >= 10.0
        nose_repr = f"silent -> {sn:.1f} Hz"
    else:
        nose_ok = dn >= 40.0
        nose_repr = f"+{dn:.0f}%"
    t1_ok = dt >= 40.0
    print(f"    nose: {nose_repr} | T1 +{dt:.0f}% | "
          f"(gate: nose fires at sharp hunger and T1 >= +40%)")
    ok = nose_ok and t1_ok
    print(f"  HUNGER gate: {'PASS' if ok else 'FAIL'}")
    return ok


def _reach_gate(brain, pools, nose_sides, vnc_sensory):
    """sugar-reach split: the INVISIBLE sugar sits on a reachable key; the
    fly's OWN antennae shift toward that side (connectome, rewired L/R), and
    the leg lands exactly on the sugar key (our key->plume->key choreography is
    `predicted` geometry). The no-false-movement assertion proves `sugar_
    should_press` yields ZERO presses when the connectome genuinely points away
    from the goal key."""
    from play_piano import (Press, _keys, _pace, capture_full_pose,
                            capture_stand_q, key_to_source, mirror, note_index,
                            settle_to_stand, CONTROL_DT, DESCEND_STEPS,
                            MIN_HOLD_S, TICK_LEN)
    naturals = "C4 D4 E4 F4 G4 A4 B4 C5 D5 E5 F5 G5 A5 B5 C6".split()
    pool = [i for i, k in enumerate(_keys()) if k["note"] in naturals]
    hues = [float(key_to_source(k)["source_y"]) for k in pool]
    cys = [float(_keys()[k]["center_xyz"][1]) for k in pool]
    mono = all(a <= b for a, b in zip(hues, hues[1:])) \
        or all(a >= b for a, b in zip(hues, hues[1:]))
    ident = all(abs(h + c) < 1e-9 for h, c in zip(hues, cys))  # sign flip exact
    print("  reach (invisible sugar ON a key; brain points, leg lands):")
    print(f"    pool: {len(pool)} naturals | key->plume offset flipped "
          f"(ok={ident}) and monotonic (ok={mono})")

    runs, seeds = {}, {"AHEAD": 5, "C4_R": 6, "C6_L": 7}
    for name, note, seed in (("AHEAD", None, 5), ("C4_R", "C4", 6),
                             ("C6_L", "C6", 7)):
        src = ({"source_x": 0.22, "source_y": 0.0} if note is None
               else key_to_source(note_index(note)))
        rec = _spatial_run(brain, pools, nose_sides, vnc_sensory,
                           (src["source_x"], src["source_y"]), ticks=30,
                           warmup=10, hunger_cmd=0.9,
                           rng=np.random.default_rng(seed))
        runs[name] = rec
        nd = float(np.mean(rec["nose_r"]) - np.mean(rec["nose_l"]))
        sy = src["source_y"]
        print(f"    [{name:5s}] sugar=scent@(0.22,{sy:+.2f}) "
              f"noseDel={nd:+.2f} fired={all(rec['firing'])}")
    base = float(np.mean(runs["AHEAD"]["nose_r"]) - np.mean(runs["AHEAD"]["nose_l"]))
    rd = float(np.mean(runs["C4_R"]["nose_r"]) - np.mean(runs["C4_R"]["nose_l"]))
    ld = float(np.mean(runs["C6_L"]["nose_r"]) - np.mean(runs["C6_L"]["nose_l"]))
    side_ok = rd > base and ld < base
    print(f"    nose vs AHEAD {base:+.2f}: C4(right) {rd:+.2f} (>), "
          f"C6(left) {ld:+.2f} (<) -> {side_ok}")

    env = mirror.build_piano_env()
    env.reset()
    a_stand = settle_to_stand(env)
    stand_q = capture_stand_q(env)
    stand_full = capture_full_pose(env)
    land = {}
    for note in ("C4", "C6"):
        ki = note_index(note)
        force = 0.3
        hold_s = max(MIN_HOLD_S, 2 * TICK_LEN * 0.5)
        hold_s = max(MIN_HOLD_S, hold_s * force)
        pr = Press(env, a_stand, ki, _pace(hold_s / CONTROL_DT), stand_q,
                   stand_full, force=force,
                   descend_steps=_pace(DESCEND_STEPS / max(force, 0.3)))
        while pr.phase != "done":
            pr.run()
        land[note] = pr.pressed_ok
        print(f"    press sugar key {note}: pressed_ok={pr.pressed_ok} "
              f"leg={pr.leg}")
    land_ok = all(land.values())
    fired_ok = all(all(r["firing"]) for r in runs.values())

    # NO-FALSE-MOVEMENT: only `sugar_should_press(known, side, want)` lets the
    # body move. The connectome GENUINELY points RIGHT on C4_R (rd heavily
    # positive) and LEFT on C6_L (ld heavily negative) — real firing. Placing
    # the sugar on the LEFT key C6 (want=-1) while the connectome only smells
    # RIGHT must return "do not press" -> literally zero presses that block.
    from play_piano import sugar_should_press
    side_r = 1.0 if rd > 0 else -1.0
    side_l = 1.0 if ld > 0 else -1.0
    miss_c6 = not sugar_should_press(True, side_r, -1.0)   # away-pointing brain
    press_r = sugar_should_press(True, side_r, 1.0)        # C4 goal, right nose
    press_l = sugar_should_press(True, side_l, -1.0)       # C6 goal, left nose
    print(f"    gate sugar_should_press: away-point (RIGHT nose, C6 goal) "
          f"-> no-press ok={miss_c6}; legit C4->press ok={press_r}; "
          f"legit C6->press ok={press_l}")
    ok = ident and mono and side_ok and fired_ok and land_ok \
        and miss_c6 and press_r and press_l
    print(f"  REACH gate: {'PASS' if ok else 'FAIL'}")
    return ok


def _asset_gate():
    have = key_sound.ensure_notes()
    missing = sorted(set(key_sound.NOTES) - set(have))
    print(f"  assets: {len(key_sound.NOTES)} C-major notes, "
          f"{len(missing)} missing: {missing or 'none'}")
    ok = not missing
    print(f"  ASSETS gate: {'PASS' if ok else 'FAIL'}")
    return ok


def main(blocks=6, block_ticks=105, warmup_ticks=35, tarsal_gain=bd.TARSAL_GAIN,
         reward_gain=bd.REWARD_GAIN):
    brain = NativeBrain(GRAPH)
    pools, nose, green, vnc_sensory, nose_sides = bd._load_pools(brain)
    tick_s = bd.TICK_MS / 1000.0
    print(f"scent_ab: {brain.n} neurons | M5 blocks={blocks} x {block_ticks}"
          f" ticks ({block_ticks*tick_s:.1f}s) + {warmup_ticks} warmup | "
          f"tarsal={tarsal_gain} reward_gain={reward_gain} | "
          f"nose L={len(nose_sides['l'])} R={len(nose_sides['r'])}", flush=True)

    t0 = time.monotonic()
    m5 = _m5_gate(brain, pools, vnc_sensory, blocks=blocks,
                  block_ticks=block_ticks, warmup_ticks=warmup_ticks,
                  tarsal_gain=tarsal_gain, reward_gain=reward_gain)
    intent = _intent_gate(brain, pools, nose_sides, vnc_sensory)
    hunger = _hunger_gate(brain, pools, nose_sides, vnc_sensory)
    assets = _asset_gate()
    reach = _reach_gate(brain, pools, nose_sides, vnc_sensory)
    wall = time.monotonic() - t0

    print(f"  wall time: {wall:.0f}s")
    ok = m5 and intent and hunger and assets and reach
    print(f"\n  ALL GATES: {'PASS' if ok else 'FAIL'} | "
          f"M5 {m5} INTENT {intent} HUNGER {hunger} ASSETS {assets} "
          f"REACH {reach}")
    return 0 if ok else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--block-ticks", type=int, default=105)
    parser.add_argument("--warmup-ticks", type=int, default=35)
    parser.add_argument("--tarsal-gain", type=float, default=bd.TARSAL_GAIN)
    parser.add_argument("--reward-gain", type=float, default=bd.REWARD_GAIN)
    args = parser.parse_args()
    sys.exit(main(blocks=args.blocks, block_ticks=args.block_ticks,
                  warmup_ticks=args.warmup_ticks,
                  tarsal_gain=args.tarsal_gain, reward_gain=args.reward_gain))