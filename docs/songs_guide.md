# Song Scoring & Keyboard Mapping Guide

This guide explains how song files are formatted, how notes map to the 25-key keyboard, and how to create custom scores for `pianist-fly`.

---

## 1. Keyboard Layout & Reachability

The 3D piano model contains **25 keys** spanning 2 chromatic octaves ($C_4$ to $C_6$).

```
[C4] C#4 [D4] D#4 [E4] [F4] F#4 [G4] G#4 [A4] A#4 [B4] [C5] C#5 [D5] D#5 [E5] [F5] F#5 [G5] G#5 [A5] A#5 [B5] [C6]
\_______________________ Right Arm (T1_R) ________________/\_______________________ Left Arm (T1_L) _________________/
```


### Reachable Set: 15 Natural Keys (White Keys)
- **Right Arm ($T1_R$, keybed $y \le 0$):** `C4`, `D4`, `E4`, `F4`, `G4`, `A4`, `B4`, `C5` (8 keys)
- **Left Arm ($T1_L$, keybed $y > 0$):** `D5`, `E5`, `F5`, `G5`, `A5`, `B5`, `C6` (7 keys)

*Note: Black accidental keys ($C\sharp, D\sharp, F\sharp, G\sharp, A\sharp$) are uncalibrated and skipped during score execution.*

---

## 2. Song File Format

Song files live in the `songs/` directory. Each line specifies a note name and duration ticks:


```text
# Line format: NOTE TICKS
# Comments starting with '#' are ignored

C4 2
E4 2
G4 4
```

### Time & Pace Tuning
- Ticks represent relative duration ($1\text{ tick} \approx 0.11\text{ s}$ at default pace).
- `FLY_PIANO_SPEED` environment variable controls global transit speed (default: `2`).

---

## 3. Built-in Scores

| Song File | Note Count | Key Range | Description |
|---|---|---|---|
| `clementi_sonatina_op36n1.txt` | 34 | $C_4 \dots G_5$ | Default score. Clementi Op.36/1 mvt I, bars 1–6 (written 1 octave down) |
| `ode_to_joy.txt` | 44 | $C_4 \dots G_4$ | Beethoven's Ode to Joy theme (Right arm only) |
| `twinkle_twinkle.txt` | 42 | $C_4 \dots A_4$ | Traditional melody (Right arm only) |
| `c_major_scale.txt` | 29 | $C_4 \dots C_6$ | Ascending and descending scale across all 15 natural keys |

---

## 4. Running Custom Scores

Pass your custom score file using `FLY_SONG`:

```bash
FLY_SONG=songs/my_song.txt ./pianist.sh
```

### Which Songs Work

**Not every song works — the fly can only reliably press the 15 natural (white) keys (`C4` … `C6`).** Keep these constraints in mind:

* **Stick to the 15 naturals:** `C4 D4 E4 F4 G4 A4 B4 C5 D5 E5 F5 G5 A5 B5 C6`. Only these are MEASURED to press (15/15 `pressed_ok` on both halves of the keybed).
* **Avoid black keys** (`C# D# F# G# A#` / flats): they exist on the 25-key model but are **uncalibrated** — presses on them are not verified and may fail. In sugar-reach mode they are skipped with a `not a reachable natural` warning.
* **Stay within `C4` … `C6`:** notes below `C4` or above `C6` are not on the keyboard and are skipped with `note ... not on keyboard, skipping`.
* **Accidentals (sharps/flats):** anything requiring `#` or `b` generally means your song won't play correctly — transcribe it to a natural-only key first (e.g. write it in C major).
* File format is `NOTE TICKS` per line; blank lines and `#` comments are ignored.

### Verified-Playable Songs

Built-in (all-natural) scores that the fly can play through completely:

| Song File | Note Count | Key Range | Notes |
|---|---|---|---|
| `clementi_sonatina_op36n1.txt` | 34 | $C_4 \dots G_5$ | Default score. Clementi Op.36/1 mvt I, bars 1–6 (written 1 octave down). Ends before bar 7's $F\sharp$ |
| `ode_to_joy.txt` | 44 | $C_4 \dots G_4$ | Beethoven's Ode to Joy theme (Right arm only) |
| `twinkle_twinkle.txt` | 42 | $C_4 \dots A_4$ | Traditional melody (Right arm only) |
| `c_major_scale.txt` | 29 | $C_4 \dots C_6$ | Ascending / descending scale over all 15 naturals |

Anything transposed into those naturals (same note names, any rhythm) works; anything needing a black key or going outside `C4..C6` will be partially skipped.

> **Driver mode (`FLY_PIANO_DRIVER`):** the note to press comes from the brain's own decision (a fixed pentatonic set `C D E G A` + `C6`), so `FLY_SONG` does **not** control which keys are pressed in that mode.
