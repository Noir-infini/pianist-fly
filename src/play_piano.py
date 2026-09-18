"""Piano choreography: the fly presses keys one after another.

Reuses the native4 standing helpers (settle_to_stand) and the Part-3 foot IK
(damped least squares + kinematic limb drive). Maps a score (default C-major
scale, or `FLY_SONG` `NOTE TICKS` lines) to key -> foot target -> leg, then
runs the per-note state machine:

    plan -> raise (straight up) -> hover over key -> descend onto key top
    -> hold (~150 ms) -> release -> emit {piano: ...} event

Leg choice: T1R/T1L for the negative/positive-y half of the keybed; the T2
legs share the outer-lateral band (|y| > 0.11) and alternate for looks.

Run (flybody-env):
  python play_piano.py                    # live, 3D viewer
  FLY_PIANO_HEADLESS=1 FLY_PIANO_SECONDS=40 python play_piano.py   # verify
  FLY_SONG=melody.txt python play_piano.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/noirinfini/fly")
import mirror  # noqa: E402
import leg_ik  # noqa: E402
import stand_native_4legs as s4  # noqa: E402
from stand_native_4legs import settle_to_stand  # noqa: E402

ACT = os.environ.get("FLY_ACTIVITY", "/tmp/malecns_activity.json")
HEADLESS = bool(os.environ.get("FLY_PIANO_HEADLESS"))
MAX_S = float(os.environ.get("FLY_PIANO_SECONDS", "0")) or None
SONG_PATH = os.environ.get("FLY_SONG", "") or None
TICK_LEN = float(os.environ.get("FLY_PIANO_TICK", "0.22"))  # s per song tick
# When set, the note to press comes from the brain: malecns_driver.py writes
# {piano_decision: {decision: "key2_E", song_i: N, ...}} into the activity file
# and play_piano presses the DECODED note instead of the local score. Keys the
# same note at most once per driver song position.
DRIVER = bool(os.environ.get("FLY_PIANO_DRIVER"))
DRIVER_KEY_NOTE = {"key0_C": "C4", "key1_D": "D4", "key2_E": "E4",
                   "key3_G": "G4", "key4_A": "A4", "key5_C2": "C6"}

CONTROL_DT = 0.002  # MuJoCo control timestep (1 env step == 1 control step)

# Transit timing in env steps (0.002 s each)
RAISE_STEPS = 70      # straight up from stance
HOVER_STEPS = 80      # translate to above the key, staying high
DESCEND_STEPS = 70    # down onto the key top
MIN_HOLD_S = 0.150    # shortest press hold
RELEASE_STEPS = 60    # back toward stance

_HERE = os.path.dirname(os.path.abspath(__file__))
_KS = None
_KEY_INDEX = None


def _keys():
    global _KS
    if _KS is None:
        _KS = json.load(open(os.path.join(
            _HERE, "piano", "key_positions.json")))["keys"]
    return _KS


def note_index(note):
    """Map a note name ("C#4", "G5" ...) to the key index or -1."""
    global _KEY_INDEX
    if _KEY_INDEX is None:
        _KEY_INDEX = {k["note"]: i for i, k in enumerate(_keys())}
    return _KEY_INDEX.get(note, -1)


# Default score: C-major scale up and back down, C4..C6.
DEFAULT_SCORE = [(n, 2) for n in (
    "C4 D4 E4 F4 G4 A4 B4 C5 D5 E5 F5 G5 A5 B5 C6".split())] + \
    [(n, 2) for n in reversed(
        "C4 D4 E4 F4 G4 A4 B4 C5 D5 E5 F5 G5 A5 B5".split())]


def load_score():
    """Return [(note, ticks), ...] from FLY_SONG or the default."""
    if not SONG_PATH:
        return list(DEFAULT_SCORE)
    out = []
    with open(SONG_PATH) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            out.append((parts[0], int(float(parts[1]))))
    return out


def driver_note():
    """The fly's decoded note from malecns_driver.py, or None (unchanged/new).

    Returns ("note", song_i) when the driver published a NEW decision, else
    (None, None) if no decision yet or the song position hasn't advanced.
    """
    if not DRIVER:
        return None, None
    global _last_driver_song_i, _last_driver_note
    try:
        with open(ACT) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None, None
    pd_ = data.get("piano_decision")
    if not isinstance(pd_, dict) or not pd_.get("decision"):
        return None, None
    note = DRIVER_KEY_NOTE.get(pd_["decision"])
    song_i = pd_.get("song_i", -1)
    if note is None or (song_i == _last_driver_song_i
                        and note == _last_driver_note):
        return None, None
    _last_driver_song_i, _last_driver_note = song_i, note
    return note, song_i


_last_driver_song_i = -1
_last_driver_note = None


def driver_motor():
    """Latest vnc_motor T1L/T1R pool rates (Hz) from malecns_driver.py."""
    try:
        with open(ACT) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return 0.0, 0.0
    m = (data.get("piano_decision") or {}).get("motor") or {}
    try:
        return float(m.get("T1L", 0.0)), float(m.get("T1R", 0.0))
    except (TypeError, ValueError):
        return 0.0, 0.0


# vnc_motor full-drive rate (task1: tarsal gain 50 → ~55 Hz pool mean) sets
# force_scale=1. The brain's own T1 side rate scales press tempo around that.
BRAIN_FORCE_FULL_HZ = 55.0
BRAIN_FORCE_LO = 0.3
BRAIN_FORCE_HI = 1.5


def brain_force(side_rate):
    """Pool rate (Hz) -> press force_scale (hold + slam-speed multiplier)."""
    if side_rate <= 0.0:
        return 1.0
    return float(np.clip(side_rate / BRAIN_FORCE_FULL_HZ, BRAIN_FORCE_LO,
                         BRAIN_FORCE_HI))


PRESS_LOG = os.environ.get("FLY_PIANO_LOG", "/tmp/piano_press_log.tsv")


def log_press(note, side, side_rate, force, pressed_ok, t):
    try:
        with open(PRESS_LOG, "a") as f:
            f.write("%s\t%s\t%.3f\t%.3f\t%d\t%.3f\n" % (
                note, side, side_rate, force, int(pressed_ok), t))
    except OSError:
        pass


def _leg_order(cy):
    """Candidate legs for a key at keyboard y; T1 is primary, T2 helps on the
    outer-lateral band."""
    if cy < 0:
        prim, alt = "T1_right", "T2_right"
    else:
        prim, alt = "T1_left", "T2_left"
    if abs(cy) > 0.11:
        return (prim, alt)
    return (prim,)


def choose_leg(env, key_idx, alt_flip):
    """Return (leg_name, q) for a leg that can press `key_idx`, or (None, None).
    T1/T2 alternate on the outer-lateral band for natural looks."""
    k = _keys()[key_idx]
    order = list(_leg_order(k["center_xyz"][1]))
    if len(order) == 2 and alt_flip % 2:
        order.reverse()
    for leg in order:
        ik = leg_ik.get_ik(env, leg)
        ok, q, _ = ik.feasible(leg_ik.press_target(key_idx))
        if ok:
            return leg, q
    return None, None


def capture_full_pose(env):
    """Record the per-limb joint qpos for ALL six legs (the puppet stance)
    right after settling, so a press can freeze everything outside the
    active limb."""
    pose = {}
    for leg in ("T1_left", "T1_right", "T2_left", "T2_right",
                "T3_left", "T3_right"):
        ik = leg_ik.get_ik(env, leg)
        pose[leg] = np.array([env.physics.data.qpos[a]
                              for a in ik.all_qadr], dtype=float)
    return pose


class Press:
    """One note's state machine; steps the env, reports pressed_ok.

    The whole body is a kinematic puppet for the duration of a press: every
    limb except the active one is pinned at the pose captured right after
    settle (STAND_FULL), and the active limb is pinned along its ramp. Root
    is pinned by ParkedTask, so the fly is rigid except the one moving limb
    -> presses are exact and deterministic, no leg can drift or wedge.
    """

    def __init__(self, env, a_stand, key_idx, hold_steps, stand_q, stand_full,
                 force=1.0, descend_steps=None):
        self.env = env
        self.a_stance = a_stand
        self.key_idx = key_idx
        self.hold_steps = int(hold_steps)
        self.stand_q = stand_q
        self.stand_full = stand_full
        self.force = float(force)
        self.desc_steps = int(DESCEND_STEPS if descend_steps is None
                              else max(1, descend_steps))
        self.phase = "plan"
        self.i = 0
        self.pressed_ok = False
        self.leg = None

        self.ik = None
        self.q0 = None
        self.q_raise = None
        self.q_hover = None
        self.q_press = None

    # -- kinematic helpers ------------------------------------------------
    def _puppet(self, q):
        """Pin every limb at its captured pose except the active one (which
        goes at `q`); zero all six limbs' dof velocities."""
        d = self.env.physics.data
        for leg, vals in self.stand_full.items():
            ik = leg_ik.get_ik(self.env, leg)
            use = self.ik.full_q(q) if leg == self.leg else vals
            for adr, v in zip(ik.all_qadr, use):
                d.qpos[adr] = float(v)
            for dof in ik.all_dof:
                d.qvel[dof] = 0.0

    def _action(self, q):
        return leg_ik.compose_action(self.ik, q, self.a_stance,
                                     release_claw=True)

    def _ramp(self, qfrom, qto, steps):
        for k in range(steps):
            q = qfrom + (k + 1) / steps * (qto - qfrom)
            self._puppet(q)
            self.env.step(self._action(q))
            self.i += 1

    def _solve(self, target, q0=None):
        ok, q, err, _ = self.ik.solve(target, q0=q0)
        return q if (ok or err <= 0.003) else None

    # -- state machine -----------------------------------------------------
    def run(self):
        if self.phase == "plan":
            self.leg, q = choose_leg(self.env, self.key_idx, 0)
            if self.leg is None:
                self.phase = "done"
                return
            self.ik = leg_ik.get_ik(self.env, self.leg)
            # canonical anchor pose (not the possibly-drifted current pose)
            self.q0 = np.array(self.stand_q[self.leg], dtype=float)
            self._puppet(self.q0)
            self.env.step(self._action(self.q0))
            k = _keys()[self.key_idx]
            tip = self.ik.d.site_xpos[self.ik.site_id].copy()
            self.q_raise = self._solve(
                [tip[0], tip[1], tip[2] + 0.040], self.q0)
            if self.q_raise is None:
                self.phase = "done"
                return
            self.q_hover = self._solve(
                [k["center_xyz"][0], k["center_xyz"][1],
                 k["key_top_z"] + 0.040], self.q_raise)
            if self.q_hover is None:
                self.phase = "done"
                return
            self.q_press = self._solve(
                leg_ik.press_target(self.key_idx), self.q_hover)
            if self.q_press is None:
                self.phase = "done"
                return
            self.phase = "raise"

        if self.phase == "raise":
            self._ramp(self.q0, self.q_raise, RAISE_STEPS)
            self.phase = "hover"
        elif self.phase == "hover":
            self._ramp(self.q_raise, self.q_hover, HOVER_STEPS)
            self.phase = "descend"
        elif self.phase == "descend":
            self._ramp(self.q_hover, self.q_press, self.desc_steps)
            self.phase = "hold"
        elif self.phase == "hold":
            for _ in range(self.hold_steps):
                self._puppet(self.q_press)
                self.env.step(self._action(self.q_press))
                self.i += 1
            pressed = mirror.piano_pressed(self.env.physics.model,
                                           self.env.physics.data)
            self.pressed_ok = self.key_idx in pressed
            self.phase = "release"
        elif self.phase == "release":
            self._ramp(self.q_press, self.q0, RELEASE_STEPS)
            self.phase = "done"

    def sim_time(self):
        return float(self.env.physics.data.time)


def _emit_piano(note, song_i, song_n, pressed_ok, t, side=None,
                side_rate=None, force=None):
    """Merge {piano: ...} into malecns_activity.json (atomic write)."""
    try:
        data = {}
        try:
            with open(ACT) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        data["piano"] = {"note": note, "song_i": song_i, "song_n": song_n,
                         "pressed_ok": pressed_ok, "t": round(t, 3)}
        if side is not None:
            data["piano"]["side"] = side
        if side_rate is not None:
            data["piano"]["side_rate_hz"] = round(side_rate, 3)
        if force is not None:
            data["piano"]["force_scale"] = round(force, 3)
        tmp = ACT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, ACT)
    except OSError:
        pass


def capture_stand_q(env):
    """Record each front leg's 6 solve-joint values right after settling; the
    canonical anchor pose for every press."""
    stand_q = {}
    for leg in ("T1_left", "T1_right", "T2_left", "T2_right"):
        ik = leg_ik.get_ik(env, leg)
        ik.write(ik.restq())
        stand_q[leg] = ik.restq()
    return stand_q


def run_piano():
    if not HEADLESS:
        import mujoco.viewer
        import mujoco

    env = mirror.build_piano_env()
    env.reset()
    a_stand = settle_to_stand(env)
    stand_q = capture_stand_q(env)
    stand_full = capture_full_pose(env)

    if HEADLESS:
        class _H:
            def is_running(self):
                return True
            def sync(self):
                pass
        viewer = _H()
    else:
        viewer = mujoco.viewer.launch_passive(
            env.physics.model.ptr, env.physics.data.ptr)
        try:
            # FREE camera (no <camera> bodies exist in the model, so FIXED
            # would point at an invalid fixedcamid and crash the viewer).
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.trackbodyid = -1
            viewer.cam.lookat = [0.16, 0.0, 0.06]
            viewer.cam.distance = 0.5
            viewer.cam.azimuth = -60.0
            viewer.cam.elevation = -22.0
        except Exception as e:
            print("(camera: %s)" % e, flush=True)

    score = load_score()
    n_notes = len(score)
    print("piano: %d notes (%s)" % (
        n_notes, SONG_PATH if SONG_PATH else "default C-major scale"),
        flush=True)

    t0 = time.perf_counter()
    passed = []
    alt_flip = 0
    song_i = 0
    while viewer.is_running() and (MAX_S is None or
                                   time.perf_counter() - t0 < MAX_S):
        if DRIVER:
            note, song_i = driver_note()
            if note is None:
                time.sleep(0.05)
                continue            # wait for the brain's next decision
            ticks = 2
        else:
            if HEADLESS and MAX_S is None and song_i >= n_notes:
                break  # one clean pass through the score, then stop
            note, ticks = score[song_i % n_notes]
        key_idx = note_index(note)
        if key_idx < 0:
            print("note %s not on keyboard, skipping" % note, flush=True)
            song_i += 1
            continue
        # Brain-force press envelope: the fly's OWN vnc_motor T1 side rate
        # (from malecns_driver.py) scales hold time and slam speed. Key half
        # picks the side (keybed y < 0 = T1R half, else T1L).
        t1l, t1r = driver_motor()
        key_cy = _keys()[key_idx]["center_xyz"][1]
        press_side = "T1_right" if key_cy < 0 else "T1_left"
        press_motor_side = "T1R" if key_cy < 0 else "T1L"
        side_rate = t1r if press_motor_side == "T1R" else t1l
        force = brain_force(side_rate)
        hold_s = max(MIN_HOLD_S, ticks * TICK_LEN * 0.5)
        hold_s = max(MIN_HOLD_S, hold_s * force)
        press = Press(env, a_stand, key_idx, hold_s / CONTROL_DT, stand_q,
                      stand_full, force=force,
                      descend_steps=int(DESCEND_STEPS / max(force, 0.3)))
        while press.phase != "done" and viewer.is_running():
            press.run()
            viewer.sync()
        passed.append(press.pressed_ok)
        up_z = float(env.physics.data.qpos[2])
        _emit_piano(note, song_i % n_notes, n_notes, press.pressed_ok,
                    press.sim_time(), side=press_side, side_rate=side_rate,
                    force=force)
        log_press(note, press_side, side_rate, force, press.pressed_ok,
                  press.sim_time())
        print("[%3d/%d] %-4s %-10s pressed_ok=%s rate=%.1fHz force=%.2f "
              "up_z=%.3f" % (
                  song_i % n_notes + 1, n_notes, note, press.leg or "?",
                  press.pressed_ok, side_rate, force, up_z), flush=True)
        # let a long note "ring" out before the next one (live only)
        if not HEADLESS:
            remain = hold_s - (RAISE_STEPS + HOVER_STEPS + DESCEND_STEPS
                               + RELEASE_STEPS) * CONTROL_DT
            if remain > 0:
                time.sleep(remain)
        song_i += 1
        alt_flip += 1

    n_ok = sum(1 for p in passed if p)
    print("piano done: %d/%d pressed_ok" % (n_ok, len(passed)), flush=True)
    if HEADLESS and n_ok < len(passed):
        print("VERIFY FAIL: expected all notes pressed in order", flush=True)
        sys.exit(2)
    print("window closed.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    run_piano()