# Neural Verification Suite & Honesty Ledger

`pianist-fly` features an automated 5-gate test suite (`scripts/scent_ab.py`) and an explicit **Honesty Ledger** categorizing measured physical facts versus modeled assumptions.


---

## 1. Automated Headless Verification Gates

Execute all gates via:
```bash
./pianist.sh verify-scent
```

| Gate Name | Objective | Acceptance Criteria | Tested Mechanism |
|---|---|---|---|
| **`M5`** | Prove scent increases motor drive | $T1$ pool rate ON vs OFF $\ge +20\%$, point-biserial $r \ge 0.5$ | Antennal injection $\rightarrow$ motor pool firing |
| **`INTENT`** | Prove directional orientation | $c_R > c_L \implies \text{Nose}_R > \text{Nose}_L$; tracks plume offset | Bilateral $12\text{L}/11\text{R}$ antenna split |
| **`HUNGER`** | Prove appetite modulation | Hungry ($H=0.9$) output beats sated ($H=0.1$) output by $> +40\%$ | Hunger boost curve ($0.4\times \rightarrow 2.4\times$) |
| **`ASSETS`** | Prove audio synthesizer integrity | All 15 C-major wavs synthesizable & non-zero byte size | Synthesizer module (`src/key_sound.py`) |
| **`REACH`** | Prove spatial hunting & gate rule | Exact key-to-plume sign flip; physical press lands on target; **no movement on wrong aim** | Absolute gate rule (`sugar_should_press`) |

---

## 2. The Honesty Ledger: Measured vs. Predicted

To preserve scientific rigor, every component is explicitly classified as **Measured** (derived empirically from physical collision or LIF spike counts) or **Predicted** (modeled parameters or robotic algorithms).

### Measured Components (Empirical Realities)
* **Kinematic Press Verification:** Verified by `mirror.piano_pressed` tip collision within $4\text{ mm}$ of keytop (`pressed_ok`).
* **Neural Firing Counts:** Actual spike outputs recorded from the 166.7k-neuron LIF simulation graph.
* **Song Score Verification:** Execution of Clementi Sonatina Op.36 No.1 ($34/34$ notes `pressed_ok`).

### Predicted Components (Model & Robotic Assumptions)
* **LIF Physics Parameters:** Membrane time constants, thresholds, and synaptic delay parameters.
* **Spatial Scent Plume:** Inverse-square concentration math ($c = 1 / (1 + (r/\sigma)^2)$).
* **Antennal Bilateral Split:** $12\text{L} / 11\text{R}$ classification based on postsynaptic partner majority.
* **Leg Trajectories & IK:** Damped Least Squares kinematic arm movements.

---

## 3. Preservation of Biological Edge Cases

### The Centerline $C_5$ Plume Case
Centerline key $C_5$ sits at keybed coordinate $y = -0.0001\text{ m}$. When the sugar plume is placed on $C_5$:
1. Plume offset is nearly $0.0\text{ m}$, yielding $c_L \approx c_R$.
2. The LIF kernel outputs equal bilateral firing rates ($\text{Nose}_L \approx \text{Nose}_R$).
3. The absolute gate treats the centerline honestly as "ahead": `brain_on_sugar` sets `want = 0` for keys inside `SUGAR_CENTER_Y_THRESH` (`play_piano.py:72`), and `sugar_should_press(known, side, want)` passes when the symmetric read `side = 0` matches — so $C_5$ presses once the antennae read level (measured in the live sugar-reach log), without injecting any lateral steering bias.

A genuinely one-sided nose is never overridden: an off-axis read still requires a matching off-axis `want`, and the no-false-movement assertion (right-pointing brain + left goal ⇒ zero presses) still holds as-is.
