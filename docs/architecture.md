# System Architecture & Technical Specification

`pianist-fly` pairs a numerical Leaky Integrate-and-Fire (LIF) simulation over the 166,700-neuron **MaleCNS connectome** with a 3D **MuJoCo fruit fly model** performing inverse-kinematics leg reaches onto a 25-key piano keyboard.

---

## 1. Process Decomposition & IPC

The application runs as three decoupled processes communicating asynchronously via atomic JSON file swaps:

```
┌───────────────────────────┐
│   src/brain_driver.py     │
│  (Python LIF Simulation)  │
│ • 166,700 MaleCNS Neurons │
│ • Scent Plume & Hunger    │
└─────────────┬─────────────┘
              │ writes atomic FLY_ACTIVITY
              ▼
   ┌────────────────────┐
   │ malecns_activity   │
   └──────┬──────┬──────┘
          │      │ reads
  reads   │      └──────────────────────────┐
          ▼                                 ▼
┌───────────────────┐             ┌───────────────────┐
│ gnat-brain-male   │             │ src/play_piano.py │
│ (Rust 3D View)    │             │ (MuJoCo 3D Body)  │
│ • 141.7k Somas    │             │ • 6-DOF Leg IK    │
│ • Live Hz Glow    │             │ • Contact Sensor  │
└───────────────────┘             └─────────┬─────────┘
                                            │ writes target plume
                                            ▼
                                  ┌───────────────────┐
                                  │   pianist_cmd     │
                                  └───────────────────┘
```

---

## 2. Neural Connectome Engine (`src/native_brain.py`)

The neural engine implements a vectorized Leaky Integrate-and-Fire (LIF) simulation over `data/graph.npz` (166,700 neurons, 25,582,938 chemical synapses).

### Key Kernel Parameters
| Parameter | Value | Description |
|---|---|---|
| $V_{\text{rest}}$ | $-52.0\text{ mV}$ | Membrane resting potential |
| $V_{\text{th}}$ | $-45.0\text{ mV}$ | Firing threshold potential |
| $\tau_m$ | $20.0\text{ ms}$ | Membrane time constant |
| $\tau_g$ | $5.0\text{ ms}$ | Synaptic conductance time constant |
| Synaptic Delay | $1.8\text{ ms}$ | Fixed propagation delay ($18$ substeps) |
| Refractory Period | $2.2\text{ ms}$ | Absolute refractory duration |
| Substep $dt$ | $0.1\text{ ms}$ | Numerical integration timestep |
| Publish Tick | $28.6\text{ ms}$ | Brain driver tick duration ($286$ substeps) |

### Algorithmic Optimization: Ring Queue Scatter
To handle 25.58M synapses in real time ($~0.3\text{--}0.5\text{ s}$ wall time per $28.6\text{ ms}$ sim tick), post-synaptic spike delivery uses an O(edges fired) ring-buffer queue:
- Synaptic delay ($1.8\text{ ms}$) is divided into $19$ ring-buffer slots.
- Firing neurons append their outbound edge IDs into future slots.
- Each substep delivers due spikes using a single `np.bincount` scatter gather over active edges, avoiding full $25.6\text{M}$ matvec operations.

---

## 3. Sensory Physics & Olfaction (`src/brain_driver.py`)

### 3.1 Spatial Sugar Plume Model
An invisible sugar target placed at keybed coordinate $y$ generates a radial concentration field sampled at the left and right antennae:
$$c(r) = \frac{1}{1 + (r / \sigma)^2}$$
- $\sigma = 0.15\text{ m}$ (Spatial plume width)
- Left antenna position: $(0.02, -0.02)\text{ m}$
- Right antenna position: $(0.02, +0.02)\text{ m}$

### 3.2 Antennal Split ($12\text{L} / 11\text{R}$)
The 23 antennal `sugar` receptor cells (sensory Index) are assigned to the left or right hemisphere based on the majority `somaSide` of their downhill post-synaptic partners:
- **Left Hemisphere:** 12 sensory cells
- **Right Hemisphere:** 11 sensory cells

### 3.3 Motivation & Hunger Dynamics
Hunger $H \in [0, 1]$ modulates antennal sensitivity:
- Rises during sugar absence ($\tau = 40\text{ s}$)
- Decays during sugar feeding ($\tau = 12\text{ s}$)
- Boost factor scales antennal drive current from $0.4\times$ (sated) to $2.4\times$ (hungry).

---

## 4. Physical Robotics & Leg Kinematics

### 4.1 MuJoCo Fly Body (`src/mirror.py`)
- Root thorax position pinned by `ParkedTask` to prevent root drift.
- Keyboard geometry loaded from `src/piano/key_positions.json`.

### 4.2 Damped Least Squares IK (`src/leg_ik.py`)

- Solves 6 DOF per leg using damped Jacobian pseudo-inverses.
- Leg motion follows 5 discrete trajectory ramps:
  $$\text{Raise (70 steps)} \rightarrow \text{Hover (80 steps)} \rightarrow \text{Descend (70 steps)} \rightarrow \text{Hold } (\ge 150\text{ ms}) \rightarrow \text{Release (60 steps)}$$

### 4.3 Key Contact Sensor (`piano_pressed`)
A key press is verified when a front claw tip site enters the key top footprint:
- Within half-extents: $\Delta x \le \text{half\_depth}$, $\Delta y \le \text{half\_width}$
- Vertical window: $|z_{\text{claw}} - z_{\text{top}}| \le 4.0\text{ mm}$
