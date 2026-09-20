#!/usr/bin/env bash
# pianist-fly: 3D MuJoCo fly body pressing the imported keyboard,
# driven live by the MaleCNS connectome LIF brain simulation.
#
# Usage:
#   ./pianist.sh          live hunting mode: the fly hunts an INVISIBLE sugar plume
#                    hopping through the song (default Clementi Sonatina Op.36 No.1).
#                    ABSOLUTE RULE: the body moves ONLY when the fly's own
#                    antennae aim at the sugar's side.
#   ./pianist.sh stop     stop all running processes
#   ./pianist.sh status   show what is running
#   ./pianist.sh headless CI verification mode
#   ./pianist.sh verify-scent run 5-gate neural verification suite
set -euo pipefail


BF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ── Runtime directories ───────────────────────────────────────────────────────
# IPC JSON files: $XDG_RUNTIME_DIR/fly/ (user-private, cleaned on logout).
BF_RUN_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}/fly-${USER:-user}/fly"
mkdir -p "$BF_RUN_DIR"

# Logs: ./logs/  (persistent, private).
BF_LOG_DIR="$BF_DIR/logs"
mkdir -p "$BF_LOG_DIR"

export FLY_ACTIVITY="$BF_RUN_DIR/malecns_activity.json"
export FLY_CMD="$BF_RUN_DIR/pianist_cmd.json"
export FLY_SOUND="${FLY_SOUND:-1}"
export FLY_PIANO_SPEED="${FLY_PIANO_SPEED:-2}"
export DISPLAY="${DISPLAY:-:0}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-1}"

BF_PY="$BF_DIR/.venv/bin/python"
BF_DRIVER="$BF_DIR/src/brain_driver.py"
BF_BODY="$BF_DIR/src/play_piano.py"
BF_BRAIN3D="$HOME/gnat/target/release/gnat-brain-male"
BF_POINTS="$BF_DIR/data/mcns_brain_points.json"

DRIVER_LOG="$BF_LOG_DIR/brain_driver.log"
PIANO_LOG="$BF_LOG_DIR/piano.log"
BRAIN3D_LOG="$BF_LOG_DIR/brain_male.log"

BF_PAT="pianist-fly/src/play_piano.py|pianist-fly/src/stand_native_4legs.py|pianist-fly/src/mirror.py|pianist-fly/src/brain_driver.py|gnat-brain-male"

stop_all() {
  pkill -f "pianist-fly/src/play_piano.py" 2>/dev/null || true
  pkill -f "pianist-fly/src/stand_native_4legs.py" 2>/dev/null || true
  pkill -f "pianist-fly/src/mirror.py" 2>/dev/null || true
  pkill -f "pianist-fly/src/brain_driver.py" 2>/dev/null || true
  pkill -f "gnat-brain-male" 2>/dev/null || true
}

MODE="${1:-sugar-reach}"
case "$MODE" in
  stop)
    echo "stopping pianist-fly…"
    stop_all
    sleep 1
    if pgrep -af "$BF_PAT" >/dev/null; then
      echo "some processes still alive:"
      pgrep -af "$BF_PAT"
    else
      echo "stopped."
    fi
    exit 0
    ;;
  status)
    pgrep -af "$BF_PAT" || echo "nothing running"
    exit 0
    ;;
  brain) ;;
  sugar-reach) export FLY_SUGAR_REACH=1 ;;
  headless) ;;
  verify-scent)
    echo "verifying headless gates: M5 / INTENT / HUNGER / ASSETS / REACH"
    echo "  (6 blocks x 3 s + warm-up + spatial + hunger + note-asset + reach"
    echo "   checks, ~15 min)"
    cd "$BF_DIR"
    "$BF_PY" "$BF_DIR/scripts/scent_ab.py" --reward-gain "${FLY_REWARD_GAIN:-1.0}"
    exit $?
    ;;
  *) echo "usage: $0 [brain|sugar-reach|headless|verify-scent|stop|status]"; exit 2 ;;
esac

# ── Preconditions ─────────────────────────────────────────────────────────────
[[ -x "$BF_PY" ]] || { echo "ERROR: $BF_PY missing (.venv not set up)"; exit 1; }
[[ -f "$BF_POINTS" ]] || {
  echo "ERROR: $BF_POINTS missing — run: $BF_DIR/.venv/bin/python $BF_DIR/scripts/build_brain_points.py"; exit 1; }
[[ -f "$BF_DRIVER" ]] || { echo "ERROR: $BF_DRIVER missing"; exit 1; }

echo "clearing any previous run…"
stop_all
sleep 1
rm -f "$FLY_ACTIVITY" "$FLY_CMD"

if [[ "$MODE" == "headless" ]]; then
  echo "[verify] headless piano (FLY_PIANO_HEADLESS=1)…"
  cd "$BF_DIR/src"
  FLY_PIANO_HEADLESS=1 FLY_PIANO_SECONDS="${FLY_PIANO_SECONDS:-40}" \
    "$BF_PY" "$BF_BODY"
  exit 0
fi

# ── Live: brain driver first — the fly's own brain ────────────────────────────
echo "[1] brain driver -> $DRIVER_LOG"
(cd "$BF_DIR/src" && setsid "$BF_PY" "$BF_DRIVER" \
    </dev/null >"$DRIVER_LOG" 2>&1 &)
echo "    loading 166,700-neuron connectome; waiting for first tick…"
ok=0
for _ in $(seq 1 180); do
  if [[ -f "$FLY_ACTIVITY" ]]; then ok=1; break; fi
  if ! pgrep -f "brain_driver.py" >/dev/null; then break; fi
  sleep 1
done
if [[ "$ok" != 1 ]]; then
  echo "ERROR: driver did not produce $FLY_ACTIVITY. Tail of $DRIVER_LOG:"
  tail -n 25 "$DRIVER_LOG" || true
  exit 1
fi

# ── 3D brain window ───────────────────────────────────────────────────────────
if [[ -x "$BF_BRAIN3D" ]]; then
  echo "[2] 3D brain window  -> $BRAIN3D_LOG"
  ( cd "$BF_DIR" && setsid "$BF_BRAIN3D" --points "$BF_POINTS" \
      --activity "$FLY_ACTIVITY" </dev/null >"$BRAIN3D_LOG" 2>&1 & )
else
  echo "[2] skipped 3D brain window: $BF_BRAIN3D not built"
  echo "    build it:  cd \"$HOME/gnat\" && cargo build --release -p gnat-brain-male"
  echo "    (optional; the fly + piano still run. wired this run by:"
  echo "     --points $BF_POINTS --activity $FLY_ACTIVITY)"
fi

# ── 3D body (sugar-reach mode) ─────────────────────────────────────────────────
if [[ "$MODE" == "sugar-reach" ]]; then
  # Discrete GPU: force GLFW onto X11/XWayland; prime-run on the Wayland
  # backend gives a broken context ("OpenGL error 0x502").
  BODY_CMD=("$BF_PY" "$BF_BODY")
  command -v prime-run >/dev/null && BODY_CMD=(prime-run "${BODY_CMD[@]}")
  echo "[3] 3D body (MuJoCo, sugar-reach) -> $PIANO_LOG"
  ( cd "$BF_DIR/src" && setsid env -u WAYLAND_DISPLAY MUJOCO_GL=glfw \
      "${BODY_CMD[@]}" </dev/null >"$PIANO_LOG" 2>&1 & )
fi

if [[ "$MODE" == "sugar-reach" ]]; then
cat <<EOF

pianist-fly: fly is hunting.
  brain driver     fly's OWN connectome smells the INVISIBLE sugar side
  3D brain window  intent firing as it aims at the sugar
  3D body (MuJoCo) the fly REACHES, ABSOLUTE RULE: it moves ONLY after its
                   own antennae aim at the key the invisible sugar sits on —
                   a miss is zero movement (sugar stays, it re-aims); on a
                   verified press the sugar hops to the song's NEXT note
                   (Clementi Sonatina default; bearing = its own firing; key
                   precision = ours; leg motion = ours)
EOF
else
cat <<EOF

pianist-fly: fly is running (brain + 3D window).
  brain driver     fly's OWN connectome, top-K somas glowing by rate each tick
  3D brain window  every soma, coloured by class, live firing on top
EOF
fi
cat <<EOF
  IPC files        $BF_RUN_DIR/
  logs             $DRIVER_LOG
                   $BRAIN3D_LOG
                   $PIANO_LOG
  stop             $BF_DIR/pianist.sh stop
EOF