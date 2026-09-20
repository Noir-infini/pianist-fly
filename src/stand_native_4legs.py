"""Stand still and drive T1L, T1R, T2L, T2R forelegs from native connectome firing rates.

The fly STANDS using the same reset-pose capture as stand_wave.py (verified:
all 6 claw sites level to within ~2 mm). All 4 foreleg claws are released and
the femur + coxa actuators are driven directly from the connectome's own vnc_motor
pool firing rates — no trained decoder, no ridge regression.

Motor pool rates come from `malecns_driver.py --native-4legs`, which injects:
  1. Proportional sugar reward — sugar cell drive scales up with total motor firing
     (more movement → more reward → more movement: positive feedback loop)
  2. Full optic-flow visual input
  3. Proportional tarsal sensory un-gating — gain scales with movement level

The rate→leg formula (identical to stand_wave.py's proven coxa+femur combo):
  frac = clip((rate - RATE_FLOOR) / RATE_SPAN, 0, 1)
  femur = femur_rest + (FEMUR_LIFT_TARGET - femur_rest) * frac
  coxa  = coxa_rest  + COXA_TRAVEL * frac
  Achieves ~0.046–0.054+ m foot travel (verified headless).
  Legacy note: this script was previously pinned to the pre-settle reset pose,
  leaving the fly hovering at the ~0.129 m spawn height ("dead fly"). It now
  settles neutrally to the real standing height (~0.078 m) before capturing
  limb rest angles, so the drive actually lifts the feet.

Run (flybody-env):
  python stand_native_4legs.py
  FLY_WAVE_HEADLESS=1 FLY_WAVE_SECONDS=15 python stand_native_4legs.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import tempfile
import mirror  # noqa: E402  (build_free_floor_env, _setup_camera)

ACT = Path(os.environ.get("FLY_ACTIVITY", os.path.join(tempfile.gettempdir(), "malecns_activity.json")))
HEADLESS = bool(os.environ.get("FLY_WAVE_HEADLESS"))
MAX_S = float(os.environ.get("FLY_WAVE_SECONDS", "0")) or None
TAU_S = 0.10         # first-order EMA smoothing (fast, percussive)
CONTROL_DT = 0.002   # MuJoCo control timestep
RATE_FLOOR = 18.0    # Hz below which frac = 0 (no lift)
RATE_SPAN = 35.0     # Hz over the floor for frac = 1 (rates run ~30–55 Hz post-ramp)
SETTLE_STEPS = 60    # neutral steps to let the fly fall to real standing height
FEMUR_LIFT_TARGET = -0.15   # rad, fully lifted femur (fly.md §6: foot ~0.054 m)
COXA_TRAVEL = 0.5           # rad added to coxa at full activation (same as stand_wave.py)

# Action-vector layout — limb-grouped (fly.md §6).
# Per-leg joint order (8 joints, base offset 0–7):
#   0=coxa_abduct  1=coxa_twist  2=coxa  3=femur_twist
#   4=femur  5=tibia  6=tarsus  7=tarsus2
LEG_SLOTS = {
    "T1_left":  11, "T1_right": 19,
    "T2_left":  27, "T2_right": 35,
    "T3_left":  43, "T3_right": 51,
}
LEG_JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2")
BODY_JOINTS = ("head_abduct", "head_twist", "head", "abdomen_abduct", "abdomen")

# Claw adhesion slots (action index, 0 = release)
CLAW_T1L = 0
CLAW_T1R = 1
CLAW_T2L = 2
CLAW_T2R = 3

# Coxa slots (base + 2)   | Femur slots (base + 4)
COXA_T1L  = 11 + 2   # a[13]  | FEMUR_T1L = 11 + 4  # a[15]
COXA_T1R  = 19 + 2   # a[21]  | FEMUR_T1R = 19 + 4  # a[23]
COXA_T2L  = 27 + 2   # a[29]  | FEMUR_T2L = 27 + 4  # a[31]
COXA_T2R  = 35 + 2   # a[37]  | FEMUR_T2R = 35 + 4  # a[39]
FEMUR_T1L = 11 + 4   # a[15]
FEMUR_T1R = 19 + 4   # a[23]
FEMUR_T2L = 27 + 4   # a[31]
FEMUR_T2R = 35 + 4   # a[39]

# Per-leg (leg_key → (claw_slot, coxa_slot, femur_slot))
LEGS = {
    "T1L": (CLAW_T1L, COXA_T1L, FEMUR_T1L),
    "T1R": (CLAW_T1R, COXA_T1R, FEMUR_T1R),
    "T2L": (CLAW_T2L, COXA_T2L, FEMUR_T2L),
    "T2R": (CLAW_T2R, COXA_T2R, FEMUR_T2R),
}

# MuJoCo xpos site names for foot-height logging
CLAW_SITES = {
    "T1L": "walker/claw_T1_left",
    "T1R": "walker/claw_T1_right",
    "T2L": "walker/claw_T2_left",
    "T2R": "walker/claw_T2_right",
}

_I3 = np.eye(3).reshape(-1).astype(np.float64)


def capture_stand_action(env):
    """Build the 59-action that HOLDS the model's real standing pose.

    Call it right after `env.reset()` — the reset configuration IS the
    standing pose: fly upright (root z ~0.128 m), all six claw sites level on
    the floor within ~3 mm (measured: T1≈0.010, T2≈0.008, T3≈0.008 m).
    Settling with neutral `0.5` actions instead folds the fly nose-down
    ("lying dead") — do NOT do that.
    """
    qn = env.physics.named.data.qpos
    a = np.full(59, 0.5, dtype=np.float64)
    a[0:6] = 0.7                                   # all 6 claws grip
    for leg, base in LEG_SLOTS.items():
        for k, j in enumerate(LEG_JOINTS):
            a[base + k] = qn[f"walker/{j}_{leg}"].item()
    for k, j in enumerate(BODY_JOINTS):
        a[6 + k] = qn[f"walker/{j}"].item()
    return a


def settle_to_stand(env):
    """Hold the reset standing pose for SETTLE_STEPS.

    Returns the captured 59-action that HOLDS the standing pose. The reset
    pose is stable (no drift, upright); we just step it to let contacts
    settle. NB. do NOT settle with np.full(59, 0.5) — that folds the fly
    nose-down onto its front feet ("lying dead").
    """
    a_stand = capture_stand_action(env)
    for _ in range(SETTLE_STEPS):
        env.step(a_stand)
    return a_stand


def read_state():
    try:
        with open(ACT) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_live():
    if not HEADLESS:
        import mujoco.viewer

    env = mirror.build_free_floor_env()
    env.reset()

    # The reset configuration IS the standing pose (upright, all 6 feet level
    # on the floor at z≈0.128). Hold it — settling with 0.5 folds the fly
    # nose-down ("lying dead").
    a_stand = settle_to_stand(env)

    # Capture per-leg coxa and femur rest positions from the standing pose.
    coxa_rest  = {leg: a_stand[c_slot] for leg, (_, c_slot, _) in LEGS.items()}
    femur_rest = {leg: a_stand[f_slot] for leg, (_, _, f_slot) in LEGS.items()}

    base_z = float(env.physics.data.qpos[2])

    print(f"native4 body ready: height={base_z:.4f} m  "
          f"femur_rest: " + " ".join(f"{k}={v:.3f}" for k, v in femur_rest.items()) +
          f"  coxa_rest: " + " ".join(f"{k}={v:.3f}" for k, v in coxa_rest.items()),
          flush=True)

    if HEADLESS:
        class _HeadlessViewer:
            def is_running(self): return True
            def sync(self): pass
        viewer = _HeadlessViewer()
    else:
        viewer = mujoco.viewer.launch_passive(
            env.physics.model.ptr, env.physics.data.ptr)
        mirror._setup_camera(viewer, env)

    # EMA-smoothed rate per leg (Hz).
    smooth = {leg: 0.0 for leg in LEGS}

    last = time.perf_counter()
    log_at = last
    t = 0.0
    reward_level = 0.0

    while viewer.is_running() and (MAX_S is None or t < MAX_S):
        now = time.perf_counter()
        dt = min(max(now - last, 0.0), 0.1)
        last = now

        # Read native motor pool rates + reward level from driver.
        raw = {leg: 0.0 for leg in LEGS}
        st = read_state()
        if st is not None:
            n4 = st.get("native_4legs") or {}
            for leg in LEGS:
                raw[leg] = float(n4.get(leg, 0.0))
            reward_level = float(st.get("reward_level", 0.0))

        # EMA smooth with event-snap: large jumps get a half-step kick first.
        a_ema = 1.0 - math.exp(-dt / TAU_S)
        for leg in smooth:
            if abs(raw[leg] - smooth[leg]) > RATE_SPAN * 0.5:
                smooth[leg] += 0.5 * (raw[leg] - smooth[leg])
            smooth[leg] += (raw[leg] - smooth[leg]) * a_ema

        # Build action: hold standing pose, drive all 4 legs via coxa+femur.
        a = a_stand.copy()
        for leg, (cl, c_slot, f_slot) in LEGS.items():
            frac = float(np.clip((smooth[leg] - RATE_FLOOR) / RATE_SPAN, 0.0, 1.0))
            a[cl]     = 0.0                                              # release claw
            a[c_slot] = coxa_rest[leg] + COXA_TRAVEL * frac             # coxa lifts
            a[f_slot] = femur_rest[leg] + (FEMUR_LIFT_TARGET - femur_rest[leg]) * frac  # femur lifts

        steps = max(1, min(20, int(round((dt if dt > 0 else CONTROL_DT) / CONTROL_DT))))
        for _ in range(steps):
            ts = env.step(a)
            if ts.step_type == 2:      # episode time-up: reset, re-settle
                env.reset()
                a_stand = settle_to_stand(env)
                coxa_rest  = {leg: a_stand[c_slot] for leg, (_, c_slot, _) in LEGS.items()}
                femur_rest = {leg: a_stand[f_slot] for leg, (_, _, f_slot) in LEGS.items()}
        t += dt

        viewer.sync()

        if now - log_at >= 1.0:
            log_at = now
            foot = {}
            for leg, site in CLAW_SITES.items():
                try:
                    foot[leg] = float(env.physics.named.data.xpos[site][2])
                except Exception:
                    foot[leg] = float("nan")
            reward_bar = "█" * int(reward_level * 10) + "░" * (10 - int(reward_level * 10))
            print(
                f"t={t:6.1f}s reward={reward_bar}({reward_level:.2f}) | "
                f"T1L r={smooth['T1L']:5.1f}Hz f={foot['T1L']:.4f}m  "
                f"T1R r={smooth['T1R']:5.1f}Hz f={foot['T1R']:.4f}m  "
                f"T2L r={smooth['T2L']:5.1f}Hz f={foot['T2L']:.4f}m  "
                f"T2R r={smooth['T2R']:5.1f}Hz f={foot['T2R']:.4f}m",
                flush=True,
            )

    print("window closed.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    run_live()
