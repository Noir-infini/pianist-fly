"""Mirror the live 2D-model brain on a 3D MuJoCo fly. Flat floor, no ball.

Reads $FLY_CMD (dance_cmd.json, written by malecns_driver.py) and converts
page's real motor accumulators) and converts the SAME signals that move
the 2D fly into 3D joint commands. No choreography, no dance routines.

  walkL/walkR -> stepping amplitude of left/right legs (tripod phase)
  walkL-walkR -> coxa-twist steering bias
  startle/flight -> crouch + fast burst
  groom -> foreleg + head oscillation
  feed -> head dip + slow foreleg motion
  quiet -> idle sway (breathing holder so it never looks dead)

Scales are adaptive (running maxima). Keys 1-4/0 inject synthetic drives
for manual testing; bridge resumes after 5s idle. Auto-resets on fall.
"""
import os
import sys

import numpy as np

from flybody.fly_envs import walk_imitation
from dm_control import composer
from dm_control.locomotion.arenas import floors
from flybody.fruitfly import fruitfly
from flybody.tasks.template_task import TemplateTask


def build_free_floor_env(time_limit=3600.0):
    """Flat floor, FREE thorax: root motion is driven kinematically from the
    same brain walk drives that move the 2D fly (position integrated from
    left/right leg drives, upright + height locked so it can't fall over).
    Legs mirror for show. No dataset, no policy, no training needed.
    """
    task = TemplateTask(
        walker=fruitfly.FruitFly,
        arena=floors.Floor(),
        force_actuators=False,
        disable_wings=True,
        joint_filter=0.01,
        adhesion_filter=0.007,
        time_limit=time_limit,
    )
    return composer.Environment(
        time_limit=time_limit, task=task,
        strip_singleton_obs_buffer_dim=True)


PIANO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piano")
KEY_BOX_HALF = 0.002  # m: half-thickness of the invisible physical key boxes


class PianoArena(floors.Floor):
    """Flat floor + the imported keyboard (prep_keyboard.py).

    The keyboard meshes (keyboard_base/white/black.obj) are attached as
    decorative geoms so the scene looks like the real thing. The KEYS
    themselves are static aabb boxes (hull-safe, unlike the flat meshes):
    one invisible-slim box per key with its top face at the exact mesh key
    top (white 0.0304, black 0.0364), colored in place of the mesh. A claw
    press onto a box top is a real MuJoCo contact -> that IS the "key press".
    """

    def _build(self):
        super()._build()
        import json as _json
        mjcf = self.mjcf_model
        kp = _json.load(open(os.path.join(PIANO_DIR, "key_positions.json")))
        for stem, rgba in (("base", "0.38 0.38 0.42 1"),
                           ("white", "0.95 0.95 0.95 1"),
                           ("black", "0.09 0.09 0.09 1")):
            mjcf.asset.add("mesh",
                           file=os.path.join(PIANO_DIR, "keyboard_%s.obj" % stem),
                           name="kb_" + stem)
            mjcf.worldbody.add("geom", type="mesh", mesh="kb_" + stem,
                               rgba=rgba, contype=0, conaffinity=0, group=1)
        for i, key in enumerate(kp["keys"]):
            cx, cy = key["center_xyz"][0], key["center_xyz"][1]
            top = key["key_top_z"]
            rgba = "0.95 0.95 0.95 1" if key["color"] == "white" else "0.05 0.05 0.05 1"
            # half_width/depth are the y/x half-extents (keytops pitch/depth).
            # Box top sits 0.5 mm ABOVE the mesh key top: avoids z-fighting with
            # the coplanar mesh face, stays inside the sensor window.
            mjcf.worldbody.add(
                "geom", name="piano_key_%02d" % i, type="box",
                size=(key["half_depth"], key["half_width"], KEY_BOX_HALF),
                pos=(cx, cy, top - KEY_BOX_HALF + 0.0005),
                rgba=rgba, contype=1, conaffinity=1)
        # The base slab must be PHYSICAL (contype=1): otherwise claws that miss
        # a key fall straight through the desk visual and end up on the floor.
        bx = kp["keybed"]["base_x"]           # x[0.095, 0.280]
        slab_cx = (bx[0] + bx[1]) / 2.0
        mjcf.worldbody.add(
            "geom", name="piano_slab", type="box",
            size=(slab_cx - bx[0], 0.1571, 0.014),
            pos=(slab_cx, 0.0, 0.014),   # top at 0.028, 2 mm under mesh top
            rgba="0.38 0.38 0.42 1", contype=1, conaffinity=1)


def build_piano_env(time_limit=3600.0):
    """Free-thorax fly parked in front of the keyboard.

    Same free walker as build_free_floor_env, rooted at the default spawn so
    the rest pose puts the front legs exactly where the keyboard was placed
    (T1 claws x~0.090 m just clear the slab near edge at 0.095; the keybed
    spans x[0.089,0.182] y[+-0.147]). The keyboard is added in PianoArena.
    """
    class ParkedTask(TemplateTask):
        """Free-thorax fly PINNED at the park pose: leg presses push against an
        immovable thorax, so IK in world coordinates stays valid mid-motion."""

        def initialize_episode(self, physics, random_state):
            super().initialize_episode(physics, random_state)
            physics.data.qpos[2] = 0.1286   # native4 standing height
            physics.data.qpos[0] = -0.005   # park 5 mm clear of the slab near edge
            self._park = physics.data.qpos[:7].copy()

        def after_step(self, physics, random_state):
            d = physics.data
            d.qpos[:7] = self._park
            d.qvel[:6] = 0.0

    task = ParkedTask(
        walker=fruitfly.FruitFly,
        arena=PianoArena(),
        force_actuators=False,
        disable_wings=True,
        joint_filter=0.01,
        adhesion_filter=0.007,
        time_limit=time_limit,
    )
    return composer.Environment(
        time_limit=time_limit, task=task,
        strip_singleton_obs_buffer_dim=True)


def run_piano_live():
    """Headful preview: fly parked in front of the keyboard, eyes on it.

    Holds the native standing pose; T1-T3 legs rest on the floor clear of the
    keybed. ESC/close exits. Ignores bridge/cmd files; the brain IN is the
    live viewer itself.
    """
    import time as _time
    import mujoco.viewer
    import stand_native_4legs as s4

    env = build_piano_env()
    env.reset()
    a_stand = s4.settle_to_stand(env)
    d = env.physics.data
    m = env.physics.model

    print("piano live: fly parked in front of the keyboard.", flush=True)

    viewer = mujoco.viewer.launch_passive(
        env.physics.model.ptr, env.physics.data.ptr)
    try:
        import mujoco
        # FREE camera: the env has no <camera> bodies, so mjCAMERA_FIXED
        # would hit an invalid fixedcamid and crash the viewer.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.trackbodyid = -1
        viewer.cam.lookat = [0.16, 0.0, 0.06]
        viewer.cam.distance = 0.45
        viewer.cam.azimuth = -60.0
        viewer.cam.elevation = -22.0
    except Exception as e:
        print("camera:", e, flush=True)

    t0 = _time.perf_counter()
    while viewer.is_running():
        env.step(a_stand)
        now = _time.perf_counter()
        if now - t0 > 2.0:
            t0 = now
            print("held stand; pressed keys: %s"
                  % sorted(piano_pressed(m, d)), flush=True)
        viewer.sync()
    print("window closed.", flush=True)
    os._exit(0)


_PRESS_KEYS = None
_PRESS_CLAWS = (
    ("walker/claw_T1_left", "T1L"), ("walker/claw_T1_right", "T1R"),
    ("walker/claw_T2_left", "T2L"), ("walker/claw_T2_right", "T2R"),
)
_PRESS_WIN = 0.004  # m: +/- vertical window around a key top that counts


def piano_pressed(model, data):
    """Return {key_index} currently held down by a front claw.

    Kinematic sensor (robust vs contact noise): a key is "pressed" while the
    tip site of a front-leg claw lies inside the key's footprint and within
    +/-_PRESS_WIN of its top face. The physical key boxes still stop a true
    press; this sensor reads intent, not mid-leg brushes.
    """
    global _PRESS_KEYS
    if _PRESS_KEYS is None:
        import json as _json
        kp = _json.load(open(os.path.join(PIANO_DIR, "key_positions.json")))
        _PRESS_KEYS = kp["keys"]
    import mujoco
    raw = getattr(model, "_model", model)
    d = data
    ret = set()
    site_x = d.site_xpos
    for site, _ in _PRESS_CLAWS:
        si = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_SITE, site)
        if si < 0:
            continue
        tip = site_x[si]
        for j, key in enumerate(_PRESS_KEYS):
            cx, cy = key["center_xyz"][0], key["center_xyz"][1]
            top = key["key_top_z"]
            # half_width is the y half-extent (across the keybed), half_depth x.
            if abs(tip[0] - cx) <= key["half_depth"] and \
               abs(tip[1] - cy) <= key["half_width"] and \
               abs(tip[2] - top) <= _PRESS_WIN:
                ret.add(j)
    return ret


# Root-travel calibration. The flybody MuJoCo model is NOT life size: its thorax
# collision semi-axis is ~4.4 cm and the body is ~20 cm long, i.e. ~40-80x a real
# 2.5 mm fly. Speeds are therefore in model metres. 0.25 m/s ~= 1.3 body-lengths/s
# (a real fly walks ~8), which reads as a clear walk without looking silly.
WALK_SPEED = float(os.environ.get("FLY_WALK_SPEED", "0.25"))  # m/s at full drive
TURN_RATE = float(os.environ.get("FLY_TURN_RATE", "1.8"))     # rad/s at full L/R imbalance
TAKEOFF_LIFT = 0.008  # hover height while behavior == 'fly'
LEASH_RADIUS = float(os.environ.get("FLY_LEASH_RADIUS", "1.5"))  # m before steering home
LEASH_RATE = 1.8      # rad/s max home-steer

LEGS = {
    "T1L": 5, "T1R": 13,
    "T2L": 21, "T2R": 29,
    "T3L": 37, "T3R": 45,
}
TRIPOD_A = ("T1L", "T2R", "T3L")
LEFT = ("T1L", "T2L", "T3L")


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def mirror_action(t, d, runmax):
    """t: seconds, d: smoothed drives dict, runmax: running maxima dict."""
    a = np.full(59, 0.5, dtype=np.float64)
    L = max(0.0, d.get("walkL", 0.0))
    R = max(0.0, d.get("walkR", 0.0))
    startle = max(0.0, d.get("startle", 0.0))
    flight = max(0.0, d.get("flight", 0.0))
    groom = max(0.0, d.get("groom", 0.0))
    feed = max(0.0, d.get("feed", 0.0))
    behavior = d.get("behavior", "idle")

    if d.get("frozen"):
        # page in background: sim halted, values stale. breathe, don't pose.
        runmax["S"] += (1.0 - runmax["S"]) * 0.05
        ph = 2 * np.pi * 0.5 * t
        for leg, base in LEGS.items():
            s = np.sin(ph + (0 if leg in TRIPOD_A else np.pi))
            a[base + 4] += 0.03 * s
            a[base + 5] += -0.03 * s
        a[4] += 0.02 * np.sin(ph)
        a[53:59] = 0.55
        return np.clip(a, 0, 1).astype(np.float32)

    # One shared scale for both sides. Independent per-side maxima saturated
    # both to 1.0, so nL - nR -> 0 and the fly never turned.
    mS = max(runmax.get("S", 1e-6), 1e-6)
    nL = _clamp(L / mS, 0.0, 1.0)
    nR = _clamp(R / mS, 0.0, 1.0)
    total = nL + nR

    # Modes come ONLY from the page's own behavior state (its thresholds are
    # calibrated to this sim). Raw magnitudes are 10-100x the code comments'
    # guesses, so absolute cutoffs would fire almost permanently.
    burst = behavior in ("startle", "fly")
    if burst:
        amp = 0.40
        ph = 2 * np.pi * 8.0 * t
        for leg, base in LEGS.items():
            s = np.sin(ph) if leg in TRIPOD_A else np.sin(ph + np.pi)
            a[base + 2] += -0.10 + 0.10 * s
            a[base + 4] += amp * s
            a[base + 5] += -amp * s
        a[4] += -0.08
        a[53:59] = 0.7
        return np.clip(a, 0, 1).astype(np.float32)

    if behavior == "groom":
        ph = 2 * np.pi * 6.0 * t
        for leg in ("T1L", "T1R"):
            base = LEGS[leg]
            a[base + 4] += 0.18 * np.sin(ph)
            a[base + 5] += -0.18 * np.sin(ph)
        a[2] += 0.12 * np.sin(ph)
        a[53:59] = 0.6
        return np.clip(a, 0, 1).astype(np.float32)

    if total < 0.06 and behavior not in ("feed",):
        # idle sway: breathing holder, clearly alive, clearly resting
        ph = 2 * np.pi * 0.5 * t
        for leg, base in LEGS.items():
            s = np.sin(ph + (0 if leg in TRIPOD_A else np.pi))
            a[base + 4] += 0.03 * s
            a[base + 5] += -0.03 * s
        a[4] += 0.02 * np.sin(ph)  # abdomen breathing
        a[2] += 0.015 * np.sin(ph * 0.5)
        a[53:59] = 0.55
        return np.clip(a, 0, 1).astype(np.float32)

    freq = 1.5 + 3.0 * _clamp(total / 2.0, 0.0, 1.0)
    ph = 2 * np.pi * freq * t
    ampL, ampR = 0.35 * nL, 0.35 * nR
    twist = _clamp((nL - nR), -1.0, 1.0) * 0.15
    for leg, base in LEGS.items():
        amp = ampL if leg in LEFT else ampR
        s = np.sin(ph) if leg in TRIPOD_A else np.sin(ph + np.pi)
        if amp > 0:
            a[base + 2] += 0.34 * amp * s
            a[base + 4] += amp * s
            a[base + 5] += -amp * s
            a[base + 6] += 0.3 * amp * s
        a[base + 1] += twist * (1 if leg in LEFT else -1)
    if behavior == "feed":
        a[2] += -0.10
        for leg in ("T1L", "T1R"):
            base = LEGS[leg]
            a[base + 4] += 0.06 * np.sin(ph)
    a[4] += 0.04 * np.sin(ph)
    a[53:59] = 0.6
    return np.clip(a, 0, 1).astype(np.float32)


KEY_DRIVES = {
    "1": {"walkL": 20.0, "walkR": 20.0, "behavior": "walk"},
    "2": {"walkL": 6.0, "walkR": 20.0, "behavior": "walk"},
    "3": {"walkL": 20.0, "walkR": 6.0, "behavior": "walk"},
    "4": {"startle": 40.0, "behavior": "startle"},
    "0": {"behavior": "rest"},
}
KEYMAP = {"1": "walk", "2": "turn-R", "3": "turn-L", "4": "burst", "0": "rest"}


def _setup_camera(viewer, env):
    """Point a position-tracking camera at the fly's root body.

    dm_control wraps MuJoCo; the raw binding is `model._model`, and flybody
    namespaces bodies as `walker/thorax`.
    """
    try:
        import mujoco
        raw = getattr(env.physics.model, "_model", env.physics.model)
        tid = -1
        for name in ("walker/thorax", "thorax"):
            tid = mujoco.mj_name2id(raw, mujoco.mjtObj.mjOBJ_BODY, name)
            if tid >= 0:
                break
        if tid < 0:
            tid = 2  # worldbody=0, fly root is the next body
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = int(tid)
        viewer.cam.distance = 0.4
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 130.0
        print(f"camera: tracking body id {tid}", flush=True)
    except Exception as e:
        print("camera setup skipped:", e, flush=True)


def run_live():
    import mujoco
    import mujoco.viewer
    import json as _json
    import os as _os
    import time as _time
    env = build_free_floor_env()  # flat floor, root travels
    env.reset()
    import math as _math
    # Let the fly settle onto its feet first. The reset pose is the spawn height
    # (~0.128 m), which leaves the feet ~5 cm above the floor; if we pin the root
    # there the fly hovers while the legs dangle ("dead fly"). The settled height
    # is the actual standing height (~0.078 m).
    for _ in range(60):
        env.step(np.full(59, 0.5, dtype=np.float64))
    base_z = float(env.physics.data.qpos[2])
    print(f"standing height: {base_z:.4f} m", flush=True)
    pos = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    # IPC path — fly.sh exports FLY_CMD into $XDG_RUNTIME_DIR/fly/;
    # fallback to /tmp/ for manual invocation without fly.sh.
    cmd_path = os.environ.get("FLY_CMD", "/tmp/dance_cmd.json")
    # NOTE: do NOT delete a pre-existing cmd file here: staleness is guarded
    # by mtime (<3s) in poll_bridge, and deleting races headless/fast loops.
    smooth = {"walkL": 0.0, "walkR": 0.0, "startle": 0.0,
              "flight": 0.0, "groom": 0.0, "feed": 0.0, "behavior": "idle"}
    runmax = {"S": 1.0}
    key = {"at": 0.0, "cmd": None}

    def keys(keycode):
        ch = chr(keycode) if 0 < keycode < 256 else ""
        if ch in KEY_DRIVES:
            key["at"] = _time.time()
            key["cmd"] = KEY_DRIVES[ch]
            # Manual drives must hit full strength instantly, regardless of
            # whatever magnitudes the driver taught runmax before (scale
            # poisoning made keypresses normalize to ~0.09 = struggling).
            drives = KEY_DRIVES[ch]
            runmax["S"] = max(1.0, float(drives.get("walkL", 0.0) or 0.0),
                               float(drives.get("walkR", 0.0) or 0.0))
            for k in ("walkL", "walkR", "startle", "flight", "groom", "feed"):
                smooth[k] = float(KEY_DRIVES[ch].get(k, 0.0) or 0.0)
            smooth["behavior"] = KEY_DRIVES[ch].get("behavior", "idle")
            smooth["frozen"] = False
            try:
                _os.remove(cmd_path)
            except OSError:
                pass
            print("manual drive:", KEYMAP[ch], flush=True)

    def poll_bridge():
        if _time.time() - key["at"] < 5.0 and key["cmd"] is not None:
            target = key["cmd"]
            live = True
        else:
            key["cmd"] = None
            try:
                st = _os.stat(cmd_path)
                if _time.time() - st.st_mtime > 3.0:
                    raise OSError  # stale: decay below, don't hold last pose
                with open(cmd_path) as f:
                    target = _json.load(f)
                live = True
            except (OSError, ValueError):
                live = False
                target = None
        if not live:
            # No fresh instruction: ease everything toward neutral instead of
            # holding a possibly-burst pose forever (stuck-burst bug).
            for k in ("walkL", "walkR", "startle", "flight", "groom", "feed"):
                smooth[k] *= 0.90
            if smooth["behavior"] not in ("rest", "sleep", "sleeping"):
                smooth["behavior"] = "idle"
            smooth["frozen"] = True
            return smooth
        a = 0.15
        for k in ("walkL", "walkR", "startle", "flight", "groom", "feed"):
            v = float(target.get(k, 0.0) or 0.0)
            smooth[k] += (v - smooth[k]) * a
        if "behavior" in target:
            smooth["behavior"] = target["behavior"]
        smooth["frozen"] = bool(target.get("frozen", False))
        if smooth["frozen"] != poll_bridge.last_frozen:
            poll_bridge.last_frozen = smooth["frozen"]
            print("stream frozen:" , smooth["frozen"], "(page hidden? keep tab visible)", flush=True)
        # ~10s recalibration. One shared scale for both sides, so L/R
        # imbalance (actual steering) survives normalisation.
        v = max(smooth["walkL"], smooth["walkR"])
        if v > runmax["S"]:
            runmax["S"] = v
        else:
            runmax["S"] += (max(v, 0.5) - runmax["S"]) * 0.02
        return smooth

    poll_bridge.last_frozen = False

    print("mirror live: brain drives -> 3D walk (floor). keys 1=walk 2/3=turn 4=burst 0=rest", flush=True)

    def drive_root(dt, t):
        # Integrate travel from the same normalized drives the joints use.
        mS = max(runmax["S"], 1e-6)
        nL = _clamp(smooth["walkL"] / mS, 0.0, 1.0)
        nR = _clamp(smooth["walkR"] / mS, 0.0, 1.0)
        beh = smooth.get("behavior", "idle")
        frozen = smooth.get("frozen", False)
        if frozen or beh in ("rest", "sleep", "sleeping", "groom"):
            v, yawrate, lift = 0.0, 0.0, 0.0
        elif beh in ("startle", "fly"):
            v, yawrate, lift = WALK_SPEED * 1.2, 0.0, TAKEOFF_LIFT if beh == "fly" else 0.0
        else:
            total = (nL + nR) / 2.0
            v = WALK_SPEED * _clamp(total, 0.0, 1.0) if total > 0.03 else 0.0
            yawrate = (nL - nR) * TURN_RATE if total > 0.03 else 0.0
            # Small gait-synced bob so the body isn't a rigid glider.
            freq = 1.5 + 3.0 * _clamp(total / 2.0, 0.0, 1.0)
            lift = 0.003 * abs(_math.sin(2 * _math.pi * freq * t)) if v > 0 else 0.0
        # Virtual leash: past LEASH_RADIUS, bias yaw home so it stays framed.
        dist = _math.hypot(pos["x"], pos["y"])
        if dist > LEASH_RADIUS and (v > 0 or yawrate != 0):
            home = _math.atan2(-pos["y"], -pos["x"])
            err = (home - pos["yaw"] + _math.pi) % (2 * _math.pi) - _math.pi
            yawrate += _clamp(err * 2.0, -LEASH_RATE, LEASH_RATE)
        pos["yaw"] += yawrate * dt
        pos["x"] += _math.cos(pos["yaw"]) * v * dt
        pos["y"] += _math.sin(pos["yaw"]) * v * dt
        drive_root.v = v
        drive_root.yawrate = yawrate
        d = env.physics.data
        d.qpos[0] = pos["x"]
        d.qpos[1] = pos["y"]
        d.qpos[2] = base_z + lift
        d.qpos[3] = _math.cos(pos["yaw"] / 2)
        d.qpos[4], d.qpos[5] = 0.0, 0.0
        d.qpos[6] = _math.sin(pos["yaw"] / 2)
        d.qvel[0] = _math.cos(pos["yaw"]) * v
        d.qvel[1] = _math.sin(pos["yaw"]) * v
        d.qvel[2] = 0.0
        d.qvel[3], d.qvel[4], d.qvel[5] = 0.0, 0.0, yawrate

    # Use launch_passive directly (not `with`) and hard-exit after the loop:
    # MuJoCo's GLFW teardown segfaults on this Wayland box when the window
    # closes, so we skip it. os._exit is intentional, not a bug.
    viewer = mujoco.viewer.launch_passive(
        env.physics.model.ptr, env.physics.data.ptr, key_callback=keys
    )
    # Follow the fly so it is always centred and big enough to see
    # (otherwise the camera sits at the world origin and the fly wanders off).
    _setup_camera(viewer, env)
    t = 0.0
    last = _time.perf_counter()
    log_at = last
    resets = 0
    for i in range(2000000):
        if not viewer.is_running():
            break
        # Real elapsed time, so WALK_SPEED is genuine m/s and the gait phase is
        # frame-rate independent. The old fixed dt=0.002 per iteration made the
        # real speed 0.015*0.002*fps ~= 2 mm/s at 60 Hz: walking in place.
        now = _time.perf_counter()
        dt = min(max(now - last, 0.0), 0.1)
        last = now
        if i % 30 == 0:
            poll_bridge()
        drive_root(dt, t)
        ts = env.step(mirror_action(t, smooth, runmax))
        t += dt
        if ts.step_type == 2:  # time-up: keep our travel pose
            env.reset()
            resets += 1
        if now - log_at >= 1.0:
            log_at = now
            dist = _math.hypot(pos["x"], pos["y"])
            print(f"t={t:6.1f}s pos=({pos['x']:+.2f},{pos['y']:+.2f}) "
                  f"yaw={pos['yaw']:+.2f} dist={dist:.2f} "
                  f"v={getattr(drive_root, 'v', 0.0):.2f} beh={smooth['behavior']}",
                  flush=True)
        viewer.sync()
    print("window closed.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    run_live()
