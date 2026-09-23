#!/usr/bin/env bash
# pianist-fly — interactive, verifiable setup.
#
# Everything the project needs from the internet is the MaleCNS connectome the
# brain IS (3 canonical Janelia .feather tables, sha256-pinned). The fly model
# (pip: flybody), the piano model (src/piano/*.obj) and all code ship in this
# repo. The 3D brain window (~/gnat, a separate Rust app) is OPTIONAL and only
# verified here — the fly + piano always run without it.
#
# Failure policy: the ONLY hard stops are a corrupted/missing feather (poisoned
# mirror protection) and the absence of a usable Python <= 3.12 (without it the
# package install cannot succeed — see step 1/7). Everything else warns and
# continues.
set -euo pipefail
cd "$(dirname "$0")"

BF_PY="$PWD/.venv/bin/python"
GNAT_BIN="${GNAT_BIN:-$HOME/gnat/target/release/gnat-brain-male}"
ACTIVITY="${FLY_ACTIVITY:-$PWD/logs/malecns_activity.json}"

ok()   { echo "  [OK]   $*"; }
warn() { echo "  [WARN] $*"; }

# Python <= 3.12 is required: dm_control's hard dependency `labmaze` ships no
# wheels for 3.13+ (pip would try a bazel source build and fail), and this
# project's stack runs on the (3,12)-or-earlier interpreters we support.
PY_VERSION_MAX="3.12"
export PY_VERSION_MAX

python_ok() {
  # $1 = interpreter path; true iff $(major, minor) <= PY_VERSION_MAX.
  "$1" -c 'import os, sys
mx = tuple(int(x) for x in os.environ["PY_VERSION_MAX"].split("."))
sys.exit(0 if sys.version_info[:2] <= mx else 1)' 2>/dev/null
}

find_python() {
  # $1 = optional explicit interpreter (env override); else scan PATH.
  if [ -n "$1" ] && python_ok "$1"; then
    printf '%s' "$1"
    return 0
  fi
  local p
  for p in python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
    if command -v "$p" >/dev/null 2>&1 && python_ok "$(command -v "$p")"; then
      printf '%s' "$(command -v "$p")"
      return 0
    fi
  done
  return 1
}

echo
echo "pianist-fly setup"
echo "  web source of truth: Janelia flyem-male-cns (3 feathers, sha256-pinned)."
echo "  fly model + piano model ship in this repo (no download)."
echo

# 1/7 ── virtualenv ────────────────────────────────────────────────────────────
echo "== 1/7  setting up .venv =="
if [ -x "$BF_PY" ]; then
    if python_ok "$BF_PY"; then
        ok ".venv already present"
    else
        warn ".venv python is $( "$BF_PY" --version 2>&1 ) — Python 3.13+ has no labmaze/ dm_control wheels"
        warn "delete .venv and re-run ./setup.sh with Python 3.12 or lower"
        ok ".venv already present"
    fi
elif FOUND_PY="$(find_python "${PYTHON3:-}")"; then
    echo "  using $( "$FOUND_PY" --version 2>&1 ) at $FOUND_PY"
    "$FOUND_PY" -m venv .venv
    ok "created .venv"
else
    echo "ERROR: Python 3.12 or lower not found." >&2
    echo "       dm_control depends on labmaze, which ships no prebuilt wheels" >&2
    echo "       for Python 3.13+ (its source build needs bazel)." >&2
    echo "       install Python <= 3.12 (e.g. apt/brew/pacman/conda/pyenv), then re-run:" >&2
    echo "           ./setup.sh" >&2
    exit 1
fi
"$BF_PY" --version

# 2/7 ── python packages ───────────────────────────────────────────────────────
echo
echo "== 2/7  installing packages (numpy, pandas, pyarrow, mujoco, dm_control, flybody) =="
"$BF_PY" -m pip install --upgrade --quiet pip
"$BF_PY" -m pip install --progress-bar on -r requirements.txt
# flybody is excluded from requirements.txt: its metadata pins numpy==1.26.4,
# which conflicts with the >=2.1 stack we run against. Its own constraint is
# stale (we import & run it fine on numpy 2.x), so install it --no-deps.
if "$BF_PY" -c "import flybody" 2>/dev/null; then
    ok "flybody already installed"
else
    echo "  installing flybody (git, --no-deps)…"
    "$BF_PY" -m pip install --progress-bar on --no-deps \
        "flybody @ git+https://github.com/TuragaLab/flybody.git"
fi
ok "packages installed"

# 3/7 ── connectome feathers ───────────────────────────────────────────────────
echo
echo "== 3/7  fetching .feather files (3 files, sha256-pinned, ~1.1 GB once) =="
echo "       a live download bar prints below; already-verified files are reused."
"$BF_PY" scripts/fetch_graph.py --no-build

# 4/7 ── compile the graph ─────────────────────────────────────────────────────
echo
echo "== 4/7  compiling graph.npz (166,700 neurons / 25.6M edges) =="
if [ ! -s data/graph.npz ]; then
    "$BF_PY" scripts/build_graph_npz.py
else
    ok "data/graph.npz already present ($(du -h data/graph.npz | cut -f1))"
fi

# 5/7 ── brain point cloud ─────────────────────────────────────────────────────
echo
echo "== 5/7  brain point cloud for the 3D brain window =="
if [ ! -s data/mcns_brain_points.json ]; then
    "$BF_PY" scripts/build_brain_points.py
else
    ok "data/mcns_brain_points.json already present ($(du -h data/mcns_brain_points.json | cut -f1))"
fi

# 6/7 ── runtime verification ──────────────────────────────────────────────────
echo
echo "== 6/7  verifying it all works (mujoco, flybody, piano model, graph) =="
if "$BF_PY" scripts/verify_runtime.py; then
    ok "all runtime checks passed"
else
    warn "some runtime checks failed — see [FAIL] above (setup continues)"
fi
echo "  booting the connectome once, headless, to prove the brain runs (~1 min)…"
BOOT_OUT="$(cd src && timeout 300 "$BF_PY" brain_driver.py --max-ticks 0 2>/dev/null || true)"
if grep -q "166700 neurons" <<<"$BOOT_OUT"; then
    ok "brain boot: $(grep -m1 '166700 neurons' <<<"$BOOT_OUT" | sed 's/^brain driver: //')"
else
    warn "brain boot did not print the expected 166,700-neuron banner"
fi

# 7/7 ── 3D brain window (~/gnat) ──────────────────────────────────────────────
echo
echo "== 7/7  verifying the 3D brain window (~/gnat, optional) =="
if [ -x "$GNAT_BIN" ]; then
    ok "gnat-brain-male built at $GNAT_BIN"
    if timeout 60 "$GNAT_BIN" --points "$PWD/data/mcns_brain_points.json" \
        --activity "$ACTIVITY" --probe >/dev/null 2>&1; then
        ok "gnat --probe rendered an offscreen frame"
    else
        warn "gnat --probe failed to render here"
    fi
    echo "       wiring: pianist.sh launches it with --points + --activity \$FLY_ACTIVITY;"
    echo "               the driver writes {tick, active:[id,n,hz]} each tick,"
    echo "               which gnat turns into brighter firing somas per tick."
else
    warn "gnat-brain-male not found — 3D brain window will be skipped by pianist.sh"
    echo "       (optional; the fly + piano always run without it)"
    echo "       build it with:  cd ~/gnat && cargo build --release -p gnat-brain-male"
fi

echo
echo "done. run:"
echo "  ./pianist.sh            live hunting mode (fly plays the piano for invisible sugar)"
echo "  ./pianist.sh verify-scent   headless 5-gate neural verification (~15 min)"
echo "  ./pianist.sh status | stop"