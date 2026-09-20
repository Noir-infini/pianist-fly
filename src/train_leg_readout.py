"""Offline reservoir readout: sugar -> a leg-lift command (fly.md Step 1).

Runs the frozen maleCNS connectome (NativeBrain) with the same procedural optic
flow the live driver uses, but a RANDOMISED sugar schedule, and records the spike
rates of a reservoir population. Then fits a ridge decoder from those rates
(optionally with a short history of past ticks) to a smoothed sugar indicator.

The expensive part is the spiking simulation (~226 ms/tick, single CPU core).
So this script separates the two:

  1. RECORD: simulate N_TICKS and cache rates/sugar to a dataset .npz.
  2. FIT:    load the cached dataset and sweep lags/lambda off it (fast, offline).

Re-running with the dataset present re-fits only; pass --retrain to simulate
again. This is an engineered readout on top of real spikes (fly.md S9). It is
not biology and involves no plasticity: the connectome weights are frozen.

    .venv/bin/python src/train_leg_readout.py
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "readout_dataset.npz"
OUT = ROOT / "data" / "leg_readout.npz"
from native_brain import NativeBrain  # ours (§NativeBrain)  # noqa: E402


TICK_MS = 28.6                      # driver tick length
N_TICKS = 2000                      # ~8 min at ~230 ms/tick
SEED = 0
TAU_S = 0.5                         # target smoothing time constant
TRAIN_FRAC = 0.70                   # time split
VAL_FRAC = 0.15                     # of the train part, for lambda selection
ACTIVE_HZ = 0.5                     # "active" reservoir threshold (train mean rate)
TOPK = 64                           # cells to light up / report
POP_GROUPS = ("descending_neuron", "vnc_motor", "cb_motor")  # fly.md S9
LAMS = np.logspace(-2, 6, 17)       # ridge lambda sweep
LAG_SWEEP = (0, 2, 4, 8, 16)        # history length in ticks (~target tau at 16)


def _optic_flow(brain, t, rng):
    """Same procedural visual world as malecns_driver.py, deterministic in t."""
    uv = brain.uv
    n = len(brain.retina)
    flow_dir = 1.0 if math.sin(t * 0.11) >= 0 else -1.0
    phase = uv[:, 0] * 12.0 - t * 3.0 * flow_dir + uv[:, 1] * 4.0
    lum = np.where(np.sin(2 * math.pi * phase) > 0, 1.0, 0.05).astype(np.float32)
    if (t % 7.0) < 0.15:
        lum[:] = 1.0
    lum += rng.normal(0, 0.03, n).astype(np.float32)
    return np.clip(lum, 0, 1)


def _next_sugar_span(rng):
    """Randomised sugar: on for 8-16 ticks, off for 16-48 ticks (fly.md S9)."""
    return int(rng.integers(8, 17)), int(rng.integers(16, 49))


def _r2_corr(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    yc, pc = y - y.mean(), yhat - yhat.mean()
    denom = math.sqrt(float(yc @ yc) * float(pc @ pc))
    corr = float(yc @ pc) / denom if denom > 0 else float("nan")
    return r2, corr


def _ridge_sweep(Ztr, ytr, Zva, yva, lams):
    """All lambdas in one eigendecomposition of the (small, PSD) Gram matrix."""
    Zm = Ztr - Ztr.mean(0)
    ym = ytr - ytr.mean()
    A = Zm.T @ Zm
    bvec = Zm.T @ ym
    s, V = np.linalg.eigh(A)
    Vb = V.T @ bvec
    Zc_va = Zva - Ztr.mean(0)
    best = (None, None, -np.inf, None)
    for lam in lams:
        w = V @ (Vb / (s + lam))
        b = ytr.mean() - Ztr.mean(0) @ w
        r2, _ = _r2_corr(yva, Zc_va @ w + b)
        if r2 > best[2]:
            best = (w, b, r2, float(lam))
    return best


def record(brain, ticks):
    """Simulate and return (pop indices, pop body ids, rates [T,P], sugar [T])."""
    sc = np.asarray(brain.superclass)
    pop = np.flatnonzero(np.isin(sc, POP_GROUPS))
    print(f"reservoir: {len(pop)} cells in {POP_GROUPS}", flush=True)

    rng = np.random.default_rng(SEED)
    sugar = np.zeros(ticks, dtype=bool)
    i = 0
    while i < ticks:
        on, off = _next_sugar_span(rng)
        sugar[i:i + on] = True
        i += on + off
    print(f"schedule: {int(sugar.sum())}/{ticks} sugar ticks, seed={SEED}", flush=True)

    rates = np.zeros((ticks, len(pop)), dtype=np.float32)
    print(f"simulating {ticks} ticks (~{ticks*0.23/60:.0f} min at ~230 ms/tick)...",
          flush=True)
    t0 = time.monotonic()
    for tick in range(ticks):
        t = tick * TICK_MS / 1000.0
        lum = _optic_flow(brain, t, np.random.default_rng(tick))
        counts, _ = brain.step(lum, TICK_MS, sugar=bool(sugar[tick]))
        rates[tick] = counts[pop] / (TICK_MS / 1000.0)
        if tick and tick % 200 == 0:
            el = time.monotonic() - t0
            print(f"  tick {tick:4d}  {el:5.1f}s  ({1000*el/tick:.0f} ms/tick)  "
                  f"spikes={int(counts.sum())}", flush=True)
    sim_s = time.monotonic() - t0
    print(f"done: {ticks} ticks in {sim_s:.1f}s ({1000*sim_s/ticks:.0f} ms/tick)",
          flush=True)
    return pop, brain.ids[pop].astype(np.int64), rates, sugar


def _smoothed_sugar(sugar):
    alpha = 1.0 - math.exp(-(TICK_MS / 1000.0) / TAU_S)
    y = np.zeros(len(sugar), dtype=np.float64)
    for k in range(1, len(sugar)):
        y[k] = y[k - 1] + alpha * (float(sugar[k]) - y[k - 1])
    return y


def build_features(Z, lags):
    """Row t = [rates[t], rates[t-1], ..., rates[t-lags]]."""
    T, P = Z.shape
    if lags == 0:
        return Z.astype(np.float64)
    X = np.empty((T - lags, P * (lags + 1)), dtype=np.float64)
    for l in range(lags + 1):
        X[:, l * P:(l + 1) * P] = Z[lags - l:T - l].astype(np.float64)
    return X


def fit_config(rates, y, y_raw, idx, lags, tag):
    X = build_features(rates[:, idx], lags)
    yt, yr = y[lags:], y_raw[lags:]
    n = len(yt)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n_train * VAL_FRAC)
    tr, va, te = slice(0, n_train - n_val), slice(n_train - n_val, n_train), slice(n_train, n)

    mu, sd = X[tr].mean(0), np.maximum(X[tr].std(0), 1e-6)
    Zs = (X - mu) / sd
    w, b, val_r2, lam = _ridge_sweep(Zs[tr], yt[tr], Zs[va], yt[va], LAMS)
    # Correct ridge shrinkage: affine-rescale the train prediction to the target
    # scale so the live clip(w.z+b, 0, 1) actually spans the leg's full travel.
    ptr = Zs[tr] @ w + b
    v = float(ptr.var())
    g = float(np.cov(ptr, yt[tr], bias=True)[0, 1] / v) if v > 0 else 1.0
    o = float(yt[tr].mean() - g * ptr.mean())
    w, b = w * g, b * g + o
    pred = Zs[te] @ w + b
    val_r2 = _r2_corr(yt[va], Zs[va] @ w + b)[0]
    r2, corr = _r2_corr(yt[te], pred)
    r2_raw, corr_raw = _r2_corr(yr[te], pred)
    # Normalisation range so the live clip((pred-lo)/(hi-lo), 0, 1) spans the
    # leg's full travel (the smoothed target itself only reaches ~0.5).
    p_lo, p_hi = (float(x) for x in np.percentile(Zs[tr] @ w + b, [5, 95]))
    print(f"[{tag}] lags={lags} n={len(idx)} d={X.shape[1]} lam={lam:g} gain={g:.2f}  "
          f"val R2={val_r2:+.3f}  test R2={r2:+.3f} corr={corr:+.3f}  "
          f"(vs raw sugar: R2={r2_raw:+.3f} corr={corr_raw:+.3f})  "
          f"cmd range [{p_lo:.3f},{p_hi:.3f}]", flush=True)
    return dict(tag=tag, lags=lags, idx=np.asarray(idx), mu=mu, sd=sd, w=w, b=b,
                P=len(idx), val_r2=val_r2, r2=r2, corr=corr, r2_raw=r2_raw,
                corr_raw=corr_raw, lam=lam, p_lo=p_lo, p_hi=p_hi)


def cell_topk(w, P, lags):
    """Per-cell importance = max |w| over its lags; return positions within idx."""
    imp = np.abs(w).reshape(lags + 1, P).max(axis=0)
    return np.argsort(-imp)[:TOPK].astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=N_TICKS)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--retrain", action="store_true", help="force re-simulation")
    args = ap.parse_args()

    t_wall0 = time.monotonic()
    if args.data.exists() and not args.retrain:
        d = np.load(args.data, allow_pickle=True)
        pop, pop_ids = d["pop"], d["pop_ids"]
        rates, sugar = d["rates"], d["sugar"].astype(bool)
        print(f"loaded dataset {args.data}  ({rates.shape[0]} ticks, "
              f"{rates.shape[1]} cells)", flush=True)
    else:
        brain = NativeBrain(ROOT / "data" / "graph.npz")
        pop, pop_ids, rates, sugar = record(brain, args.ticks)
        np.savez(args.data, pop=pop.astype(np.int32), pop_ids=pop_ids,
                 rates=rates, sugar=sugar,
                 groups=np.array(",".join(POP_GROUPS)),
                 tick_ms=np.float32(TICK_MS), seed=np.int32(SEED))
        print(f"cached dataset -> {args.data}  "
              f"({args.data.stat().st_size/1e6:.1f} MB)", flush=True)

    y = _smoothed_sugar(sugar)
    y_raw = sugar.astype(np.float64)
    n_train = int(len(sugar) * TRAIN_FRAC)
    act_sel = np.flatnonzero(rates[:n_train].mean(0) > ACTIVE_HZ)
    print(f"active subset (> {ACTIVE_HZ} Hz on train): {len(act_sel)} / {len(pop)}",
          flush=True)

    configs = [fit_config(rates, y, y_raw, np.arange(len(pop)), 0, "all")]
    configs += [fit_config(rates, y, y_raw, act_sel, l, "active") for l in LAG_SWEEP]
    best = max(configs, key=lambda c: c["val_r2"])
    print(f"selected {best['tag']} lags={best['lags']} by validation R2", flush=True)

    P = best["P"]
    topk = cell_topk(best["w"], P, best["lags"])
    np.savez(
        args.out,
        idx=pop[best["idx"]].astype(np.int32),
        pop_ids=pop_ids[best["idx"]].astype(np.int64),
        mu=best["mu"].astype(np.float32), sd=best["sd"].astype(np.float32),
        w=best["w"].astype(np.float32), b=np.float32(best["b"]),
        p_lo=np.float32(best["p_lo"]), p_hi=np.float32(best["p_hi"]),
        topk=topk, lags=np.int32(best["lags"]),
        r2=np.float32(best["r2"]), corr=np.float32(best["corr"]),
        val_r2=np.float32(best["val_r2"]),
        choice=np.array(best["tag"]), seed=np.int32(SEED), n_ticks=np.int32(len(sugar)),
        tau_s=np.float32(TAU_S), lam=np.float32(best["lam"]),
        active_hz=np.float32(ACTIVE_HZ), groups=np.array(",".join(POP_GROUPS)),
        topk_ids=pop_ids[best["idx"]][topk],
    )
    print(f"saved {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)  "
          f"lags={best['lags']}", flush=True)
    print("top readout cell body IDs:",
          [int(x) for x in pop_ids[best["idx"]][topk][:16]], flush=True)
    print(f"total wall time {time.monotonic()-t_wall0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
