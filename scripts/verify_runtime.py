#!/usr/bin/env python3
"""Runtime verification for the pianist-fly install (setup.sh step 6).

Headless checks that "everything actually works" without opening a window:

  1. python venv + library versions (numpy, pandas, pyarrow, mujoco, flybody)
  2. compiled connectome loads (166,700 neurons / 25,582,938 edges)
  3. piano model assets present + all 25 keys decoded
  4. MuJoCo builds the fly + keyboard scene headlessly and steps (no GL)

Exit code 0 = all passed; nonzero = at least one [FAIL]. setup.sh treats the
result as warn-and-continue (only the pinned feather hashes are hard gating).
"""
import json
import sys
import warnings
from pathlib import Path

# dm_control can warn its internals on numpy >= 2.5 on import/reset; not ours.
warnings.filterwarnings("ignore", module=r"dm_control\.mujoco\.index")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PIANO = SRC / "piano"

GRAPH = ROOT / "data" / "graph.npz"
EXPECT_N = 166_700
EXPECT_E = 25_582_938
EXPECT_KEYS = 25
PIANO_ASSETS = ["keyboard_base.obj", "keyboard_white.obj",
                "keyboard_black.obj", "keyboard_clean.obj",
                "key_positions.json"]


def ok(msg):
    print("  [OK]   " + msg, flush=True)


def fail(msg):
    print("  [FAIL] " + msg, flush=True)


def verify(graph=GRAPH):
    checks = passed = 0
    def check(good, msg):
        nonlocal checks, passed
        checks += 1
        if good:
            passed += 1
            ok(msg)
        else:
            fail(msg)

    # 1 — libraries
    try:
        import numpy
        import pandas
        import pyarrow
        import mujoco
        import dm_control
        import flybody  # noqa: F401  (the fly model)
        from importlib import metadata as _md
        dm_v = _md.version("dm_control")
        check(True, f"libs: numpy {numpy.__version__} | pandas {pandas.__version__} "
                    f"| pyarrow {pyarrow.__version__}")
        check(True, f"libs: mujoco {mujoco.__version__} | dm_control {dm_v} "
                    f"| flybody OK")
    except Exception as e:
        check(False, f"dependency import failed: {e}")

    # 2 — compiled connectome
    try:
        import numpy
        d = numpy.load(graph, allow_pickle=True)
        n = int(len(d["ids"]))
        e = int(d["ptr"][-1])
        check(n == EXPECT_N and e == EXPECT_E,
              f"graph.npz loads: {n} neurons, {e:,} edges "
              f"(expected {EXPECT_N} / {EXPECT_E:,})")
    except Exception as ex:
        check(False, f"graph.npz load failed: {ex}")

    # 3 — piano model assets
    missing = [f for f in PIANO_ASSETS if not (PIANO / f).exists()]
    check(not missing, f"piano assets present ({len(PIANO_ASSETS) - len(missing)}"
                       f"/{len(PIANO_ASSETS)}"
                       + (f"; missing: {', '.join(missing)})" if missing else ")"))
    try:
        kp = json.loads((PIANO / "key_positions.json").read_text())
        nkeys = len(kp.get("keys", []))
        check(nkeys == EXPECT_KEYS, f"key_positions.json decodes {nkeys} keys "
                                    f"(expected {EXPECT_KEYS})")
    except Exception as ex:
        check(False, f"key_positions.json unreadable: {ex}")

    # 4 — MuJoCo builds the fly + keyboard scene headlessly and steps
    sys.path.insert(0, str(SRC))
    try:
        import numpy
        import mirror
        warnings.filterwarnings("ignore", module=r"dm_control\.mujoco\.index")
        env = mirror.build_piano_env()
        env.reset()
        ph = env.physics
        for _ in range(5):
            ph.step()
        nan = not (numpy.isfinite(ph.data.qpos).all() and numpy.isfinite(ph.data.qvel).all())
        check(not nan,
              f"MuJoCo fly+keyboard model assembled headless and steps "
              f"({ph.model.ptr.nq} qpos, {ph.model.ptr.nu} actuators, 0 NaNs)")
    except Exception as ex:
        check(False, f"headless MuJoCo fly+keyboard build failed: {ex}")

    print(f"\n{passed}/{checks} checks passed.", flush=True)
    return 0 if passed == checks else 1


if __name__ == "__main__":
    sys.exit(verify())