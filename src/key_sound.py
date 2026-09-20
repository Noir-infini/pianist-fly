"""key_sound — the piano the fly hears.

Every verified key press rings on the host speaker. The notes of the
C-major scale score (C4..C6, 15 naturals) are synthesized as a small
piano-ish pluck (fundamental + two harmonics with an exponential
ring-out) into data/notes/ as mono 16-bit wavs, then played live with
`paplay` (pulse → HDA Analog) at a volume scaled by the brain-set press
force — a hungrier fly presses harder and rings louder.

The sound is ALWAYS the fly's own verified press: nothing is played unless
play_piano.py confirmed pressed_ok. Headless/verify runs never emit sound,
and a missing player is a soft no-op (warned once).

Run (pre-synthesize):  .venv/bin/python src/key_sound.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
NOTES_DIR = _HERE.parent / "data" / "notes"

SR = 44100
DUR_S = 1.2
DECAY_TAU_S = 0.5
ATTACK_S = 0.008
HARMONICS = ((1, 1.0), (2, 0.5), (3, 0.25))

# The C-major scale score spans these 15 natural keys (C4..C6).
NOTES = {
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00,
    "A4": 440.00, "B4": 493.88,
    "C5": 523.25, "D5": 587.33, "E5": 659.26, "F5": 698.46, "G5": 783.99,
    "A5": 880.00, "B5": 987.77,
    "C6": 1046.50,
}

_PLAY = shutil.which("paplay") or shutil.which("pw-play")
_WARNED = False


def _render(freq: float) -> np.ndarray:
    t = np.linspace(0.0, DUR_S, int(SR * DUR_S), endpoint=False)
    env = np.exp(-t / DECAY_TAU_S) * np.clip(t / ATTACK_S, 0.0, 1.0)
    tone = sum(a * np.sin(2.0 * np.pi * m * freq * t)
               for m, a in HARMONICS)
    y = tone * env
    y = 0.9 * y / (max(1e-9, float(np.max(np.abs(y)))))
    return (y * 32767.0).astype(np.int16)


def _write_wav(name: str, samples: np.ndarray) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    with wave.open(str(NOTES_DIR / (name + ".wav")), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


def ensure_notes(force: bool = False):
    """Synthesize the wavs for the score's 15 naturals; return the set that
    exists after (re)writing. Cheap: ~46 KB per note, ~415 ms of math total.
    """
    existing = {p.stem for p in NOTES_DIR.glob("*.wav")}
    if force:
        existing.clear()
    for name, freq in NOTES.items():
        if name not in existing:
            _write_wav(name, _render(freq))
            existing.add(name)
    return existing


def play_note(name: str, force: float = 1.0) -> None:
    """Play one verified key press at brain-force-scaled volume (0.3..1.5).

    Non-blocking. Missing note files are synthesized lazily; a missing audio
    player is a soft no-op so headless/science runs never break.
    """
    global _WARNED
    if name not in NOTES:
        return
    if _PLAY is None:
        if not _WARNED:
            print("key_sound: paplay/pw-play not found; audio muted",
                  flush=True)
            _WARNED = True
        return
    ensure_notes()
    f = float(force)
    pct = 0.50 + 0.50 * float(np.clip((f - 0.3) / max(1.5 - 0.3, 1e-9), 0.0, 1.0))
    wav = str(NOTES_DIR / (name + ".wav"))
    if _PLAY.endswith("paplay"):
        subprocess.Popen([_PLAY, "--volume", str(int(pct * 65536)), wav],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen([_PLAY, wav],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    got = ensure_notes(force=True)
    print(f"key_sound: {len(got)} note wavs in {NOTES_DIR}")
    for n in sorted(NOTES):
        print(f"  {n}: {n} Hz" + ("" if n in got else " (MISSING)"))