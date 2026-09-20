"""Piano choreography: the fly presses keys one after another.

Reuses the native4 standing helpers (settle_to_stand) and the Part-3 foot IK
(damped least squares + kinematic limb drive). Maps a score (default Clementi
Sonatina Op.36 No.1 mvt I, or `FLY_SONG` `NOTE TICKS` lines) to key -> foot
target -> leg, then
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
import mirror  # noqa: E402
import leg_ik  # noqa: E402
import stand_native_4legs as s4  # noqa: E402
from stand_native_4legs import settle_to_stand  # noqa: E402

import tempfile
TMP = tempfile.gettempdir()
ACT = os.environ.get("FLY_ACTIVITY", os.path.join(TMP, "malecns_activity.json"))
CMD = os.environ.get("FLY_CMD", os.path.join(TMP, "pianist_cmd.json"))

HEADLESS = bool(os.environ.get("FLY_PIANO_HEADLESS"))
MAX_S = float(os.environ.get("FLY_PIANO_SECONDS", "0")) or None
SONG_PATH = os.environ.get("FLY_SONG", "") or None
TICK_LEN = float(os.environ.get("FLY_PIANO_TICK", "0.22"))  # s per song tick
# When set, brain_driver.py writes {piano_decision: ...} into the activity
# file and we press the DECODED note instead of the local score (each song
# position at most once).
DRIVER = bool(os.environ.get("FLY_PIANO_DRIVER"))
DRIVER_KEY_NOTE = {"key0_C": "C4", "key1_D": "D4", "key2_E": "E4",
                   "key3_G": "G4", "key4_A": "A4", "key5_C2": "C6"}
# Sugar-reach mode: an INVISIBLE sugar source sits on a reachable key; we write
# its position into FLY_CMD so the driver's plume points there, the fly's own
# antennae fire toward it, and we PRESS exactly that key. On a VERIFIED press
# the sugar hops to the song's NEXT note (melody order). FLY_SUGAR_HOPS bounds
# a headless run (0 = one clean pass).
SUGAR_REACH = bool(os.environ.get("FLY_SUGAR_REACH"))
SUGAR_HOPS = int(os.environ.get("FLY_SUGAR_HOPS", "0")) or None
SUGAR_STRENGTH = 0.8
SUGAR_SOURCE_X = 0.22          # plume ahead distance, mirrors the INTENT gate
# ABSOLUTE gate: the leg moves ONLY when the connectome's own L/R firing
# points at the placed sugar's side; a miss leaves the sugar put and the fly
# re-aims. SUGAR_RETRIES bounds miss polls (0 = wait forever, live default).
SUGAR_RETRIES = int(os.environ.get(
    "FLY_SUGAR_RETRIES",
    "40" if HEADLESS else "0")) or None
SUGAR_POLL_S = 0.25
# Keys within SUGAR_CENTER_Y_THRESH of the centreline (world y) read as "ahead"
# (want=0.0), matching the brain's symmetric side=0.0. Only C5 (y=−0.0001)
# falls inside the band.
SUGAR_CENTER_Y_THRESH = 0.005

CONTROL_DT = 0.002  # MuJoCo control timestep (1 env step == 1 control step)

# Pace knob: FLY_PIANO_SPEED divides every press transit step count (higher
# = faster, default 2x). Endpoints never change, so verification still holds;
# the floor keeps a kinematic ramp from teleporting to the next target.
PACER = max(1.0, float(os.environ.get("FLY_PIANO_SPEED", "2")))


def _pace(n):
    return max(6, int(round(n / PACER)))

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
        with open(os.path.join(_HERE, "piano", "key_positions.json")) as _f:
            _KS = json.load(_f)["keys"]
    return _KS


def note_index(note):
    """Map a note name ("C#4", "G5" ...) to the key index or -1."""
    global _KEY_INDEX
    if _KEY_INDEX is None:
        _KEY_INDEX = {k["note"]: i for i, k in enumerate(_keys())}
    return _KEY_INDEX.get(note, -1)


# Default score: Clementi Sonatina Op.36 No.1 mvt I opening theme (C major,
# C4..G5, bars 1-6; printed RH is written one octave down to fit C4..C6).
# Loaded straight from songs/clementi_sonatina_op36n1.txt — the file is the
# single source of truth. Parse exactly like load_score() does for FLY_SONG.
def _read_song_file(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            out.append((parts[0], int(float(parts[1]))))
    return out


DEFAULT_SCORE = _read_song_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "songs", "clementi_sonatina_op36n1.txt"))


def load_score():
    """Return [(note, ticks), ...] from FLY_SONG or the default."""
    if not SONG_PATH:
        return list(DEFAULT_SCORE)
    # The body runs from src/, so resolve relative FLY_SONG against the repo
    # root too (e.g. "songs/ode_to_joy.txt"); else open() fails with the path.
    path = SONG_PATH
    if not os.path.isfile(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, path)
        if os.path.isfile(alt):
            path = alt
    out = []
    with open(path) as f:
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
    """The fly's decoded note from brain_driver.py, or None (unchanged/new).

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
    """Latest vnc_motor T1L/T1R pool rates (Hz) from brain_driver.py."""
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


# Full-drive vnc_motor pool rate (~55 Hz) maps to force_scale=1; the brain's
# own T1 side rate scales press tempo around that.
BRAIN_FORCE_FULL_HZ = 55.0
BRAIN_FORCE_LO = 0.3
BRAIN_FORCE_HI = 1.5


def brain_force(side_rate):
    """Pool rate (Hz) -> press force_scale (hold + slam-speed multiplier)."""
    if side_rate <= 0.0:
        return 1.0
    return float(np.clip(side_rate / BRAIN_FORCE_FULL_HZ, BRAIN_FORCE_LO,
                         BRAIN_FORCE_HI))


# ---- sugar-reach: an invisible sugar on a key, and the fly reaching it ------
# Reachable set = the 15 naturals, each MEASURED to press (15/15 pressed_ok on
# both halves of the keybed).
SUGAR_NATURALS = ("C4 D4 E4 F4 G4 A4 B4 C5 D5 E5 F5 G5 A5 B5 C6".split())


def _sugar_pool():
    """key_idx of every reachable natural, in keybed order (C4..C6)."""
    return [i for i, k in enumerate(_keys()) if k["note"] in SUGAR_NATURALS]


def key_to_source(key_idx):
    """Invisible plume placement (`predicted`): the key's own keyboard Y becomes
    the off-axis plume offset. Sign flips because keybed y<0 is the fly's RIGHT
    half while the driver's +y is the right-antenn side; keybed y=-0.137 (C4)
    -> source_y=+0.137, the same magnitude as the INTENT-gate RIGHT source."""
    cy = _keys()[key_idx]["center_xyz"][1]
    return {"scent": True, "scent_strength": SUGAR_STRENGTH,
            "source_x": SUGAR_SOURCE_X, "source_y": float(-cy)}


def _write_sugar_cmd(key_idx):
    try:
        tmp = CMD + ".tmp"
        with open(tmp, "w") as f:
            json.dump(key_to_source(key_idx), f)
        os.replace(tmp, CMD)
    except OSError as e:
        print("sugar cmd write: %s" % e, flush=True)


def sugar_should_press(known, side, want):
    """ABSOLUTE gate: the body moves ONLY when the connectome itself points at
    the side the sugar is placed on. Every other read -> ZERO movement."""
    return bool(known) and side == want


def brain_on_sugar(key_idx):
    """The latest brain read: did the connectome's own antennae pick the side
    the sugar was placed on? Returns a dict (never raises)."""
    cy = _keys()[key_idx]["center_xyz"][1]
    # Three-state want: -1 left | 0 ahead | +1 right. C5's plume (y≈±0.0001)
    # gives the connectome cR≈cL → side=0.0; the old binary cy<0 check read
    # want=1.0, so 0.0≠1.0 stalled forever. Keys inside SUGAR_CENTER_Y_THRESH
    # now get want=0.0 to match the brain's symmetric read.
    want = (0.0 if abs(cy) < SUGAR_CENTER_Y_THRESH
            else (1.0 if cy < 0.0 else -1.0))  # keybed right half == driver RIGHT
    try:
        with open(ACT) as f:
            data = json.load(f)
        s = data.get("senses", {}).get("scent", {}) or {}
        it = data.get("intent", {}) or {}
        cL = float(s.get("cL", 0.0))
        cR = float(s.get("cR", 0.0))
        side = (1.0 if cR - cL > 1e-9 else (-1.0 if cR - cL < -1e-9 else 0.0))
        return {"aware": sugar_should_press(it.get("known"), side, want),
                "known": bool(it.get("known")), "firing": bool(it.get("firing")),
                "cL": cL, "cR": cR, "side": side, "want": want,
                "bearing": float(it.get("bearing_deg", 0.0))}
    except (OSError, ValueError, TypeError):
        return {"aware": False, "known": False, "firing": False,
                "cL": 0.0, "cR": 0.0, "side": 0.0, "want": want,
                "bearing": 0.0}


def retire_wait(key_idx):
    """Re-aim until the connectome itself aims at the placed sugar. Every miss
    is logged honestly and NOTHING moves. Returns the first aware read, or None
    once SUGAR_RETRIES consecutive polls miss (0/None = wait forever)."""
    n = 0
    while True:
        read = brain_on_sugar(key_idx)
        if read["aware"]:
            return read
        n += 1
        if SUGAR_RETRIES is not None and n > SUGAR_RETRIES:
            return None
        print("[reach] brain missed %s side -> not moving (re-aim %d; "
              "known=%s noseL=%.2f noseR=%.2f want=%+.0f)" % (
                  _keys()[key_idx]["note"], n, read["known"],
                  read["cL"], read["cR"], read["want"]), flush=True)
        time.sleep(SUGAR_POLL_S)


PRESS_LOG = os.environ.get("FLY_PIANO_LOG", os.path.join(TMP, "piano_press_log.tsv"))



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
                              else max(1, _pace(descend_steps)))
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
            self._ramp(self.q0, self.q_raise, _pace(RAISE_STEPS))
            self.phase = "hover"
        elif self.phase == "hover":
            self._ramp(self.q_raise, self.q_hover, _pace(HOVER_STEPS))
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
            self._ramp(self.q_press, self.q0, _pace(RELEASE_STEPS))
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
            env.physics.model.ptr, env.physics.data.ptr,
            show_left_ui=False, show_right_ui=False)
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

    if SUGAR_REACH and DRIVER:
        raise SystemExit("FLY_SUGAR_REACH and FLY_PIANO_DRIVER are exclusive")
    if SUGAR_REACH and HEADLESS and MAX_S is None \
            and os.environ.get("FLY_SUGAR_RETRIES", "40") == "0":
        raise SystemExit("sugar-reach headless with FLY_SUGAR_RETRIES=0 has "
                         "no bound (set FLY_SUGAR_HOPS/FLY_PIANO_SECONDS)")

    score = load_score()
    n_notes = len(score)
    if SUGAR_REACH:
        pool_keys = set(_sugar_pool())
        # The sugar follows the SONG in melody order (song file = authority),
        # keyed to the naturals the fly is MEASURED to reach.
        reach_notes = []
        for note, ticks in score:
            ki = note_index(note)
            if ki in pool_keys:
                reach_notes.append((note, ki, ticks))
            else:
                print("sugar-reach: %s not a reachable natural, skipping"
                      % note, flush=True)
        if not reach_notes:
            raise SystemExit("sugar-reach: song has no reachable naturals")
        reached = 0
        reach_i = 0
        sugar_key = reach_notes[0][1]
        _write_sugar_cmd(sugar_key)
        print("piano: sugar-reach over '%s' (%d melody notes; the invisible "
              "sugar hops to the next note after each VERIFIED brain-aimed "
              "press)" % (SONG_PATH or "default Clementi Sonatina Op.36 No.1 (C)",
                         len(reach_notes)),
              flush=True)
    else:
        print("piano: %d notes (%s)" % (
            n_notes, SONG_PATH if SONG_PATH else "default Clementi Sonatina Op.36 No.1 (C)"),
            flush=True)

    t0 = time.perf_counter()
    passed = []
    alt_flip = 0
    song_i = 0
    while viewer.is_running() and (MAX_S is None or
                                   time.perf_counter() - t0 < MAX_S):
        if SUGAR_REACH:
            if SUGAR_HOPS is not None and reached >= SUGAR_HOPS:
                break
            if HEADLESS and MAX_S is None and reach_i >= len(reach_notes):
                break  # one clean pass through the song, then stop
            # ABSOLUTE gate: no press, no hop until the connectome's own L/R
            # firing points at the placed sugar's side; a miss re-aims.
            brain = retire_wait(sugar_key)
            if brain is None:
                print("[reach] FAIL: brain never aimed at %s -> not moving"
                      % _keys()[sugar_key]["note"], flush=True)
                if HEADLESS:
                    print("VERIFY FAIL: no press without the brain's direction",
                          flush=True)
                    sys.exit(2)
                continue
            note, sugar_key, ticks = reach_notes[reach_i % len(reach_notes)]
        elif DRIVER:
            note, song_i = driver_note()
            if note is None:
                time.sleep(0.05)
                continue            # wait for the brain's next decision
            ticks = 2
        else:
            if HEADLESS and MAX_S is None and song_i >= n_notes:
                break  # one clean pass through the score, then stop
            note, ticks = score[song_i % n_notes]
        key_idx = sugar_key if SUGAR_REACH else note_index(note)
        if key_idx < 0:
            print("note %s not on keyboard, skipping" % note, flush=True)
            song_i += 1
            continue
        # Brain-force envelope: the fly's OWN vnc_motor T1 side rate scales hold
        # time and slam speed; key half picks the side (y<0 = T1R half).
        t1l, t1r = driver_motor()
        key_cy = _keys()[key_idx]["center_xyz"][1]
        press_side = "T1_right" if key_cy < 0 else "T1_left"
        press_motor_side = "T1R" if key_cy < 0 else "T1L"
        side_rate = t1r if press_motor_side == "T1R" else t1l
        force = brain_force(side_rate)
        hold_s = max(MIN_HOLD_S, ticks * TICK_LEN * 0.5)
        hold_s = max(MIN_HOLD_S, hold_s * force)
        press = Press(env, a_stand, key_idx, _pace(hold_s / CONTROL_DT),
                      stand_q, stand_full, force=force,
                      descend_steps=_pace(DESCEND_STEPS
                                          / max(force, 0.3)))
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
        # A VERIFIED press rings on the host speaker at brain-force-scaled
        # volume (a hungrier fly presses harder -> louder).
        if not HEADLESS and os.environ.get("FLY_SOUND", "1") != "0" \
                and press.pressed_ok:
            try:
                from key_sound import play_note
                play_note(note, force)
            except Exception as e:
                print("key_sound: %s" % e, flush=True)
        if SUGAR_REACH:
            # seek = whether the connectome's antennae aimed at our side
            seek = "on " if brain["aware"] else "miss"
            print("[reach %2d/%d] %-4s leg=%-9s pressed_ok=%s brain=%s "
                  "noseL=%.2f noseR=%.2f want=%+.0f deg=%+.1f" % (
                      reached + 1, len(reach_notes), note, press.leg or "?",
                      press.pressed_ok, seek, brain["cL"], brain["cR"],
                      brain["want"], brain["bearing"]), flush=True)
            if press.pressed_ok:
                reached += 1
                # hop the sugar to the song's NEXT note (melody order)
                reach_i += 1
                sugar_key = reach_notes[reach_i % len(reach_notes)][1]
                _write_sugar_cmd(sugar_key)
        else:
            print("[%3d/%d] %-4s %-10s pressed_ok=%s rate=%.1fHz force=%.2f "
                  "up_z=%.3f" % (
                      song_i % n_notes + 1, n_notes, note, press.leg or "?",
                      press.pressed_ok, side_rate, force, up_z), flush=True)
        # let a long note "ring" out before the next one (live only)
        if not HEADLESS:
            remain = hold_s - (_pace(RAISE_STEPS) + _pace(HOVER_STEPS)
                               + _pace(DESCEND_STEPS) + _pace(RELEASE_STEPS)) \
                * CONTROL_DT
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