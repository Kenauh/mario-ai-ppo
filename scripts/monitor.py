"""Monitor de una corrida: python scripts/monitor.py <name> [--window 100000]"""
import argparse
import os
import pandas as pd

BASELINE_CSV = "results/baseline_random.csv"
GATE_STEPS, GATE_MEAN, GATE_DET = 1_000_000, 1000, 700


def read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def baseline_mean():
    df = read(BASELINE_CSV)
    if df is None or "max_x" not in df.columns:
        return None
    return float(df["max_x"].mean())


def gate(ep, ev, gs):
    print(f"--- Puerta de decision a {gs} pasos ---")
    w = ep[(ep["timesteps"] > gs - 100000) & (ep["timesteps"] <= gs)]
    m = float(w["max_x"].mean()) if len(w) else float("nan")
    ok_m = len(w) > 0 and m >= GATE_MEAN
    ok_d, d = False, float("nan")
    if ev is not None and len(ev):
        row = ev.iloc[(ev["timesteps"] - gs).abs().argmin()]
        d = float(row["det_max_x"])
        ok_d = d > GATE_DET
        print(f"Evaluacion determinista mas cercana: {int(row['timesteps'])} pasos, det_max_x={d:.0f}")
    print(f"Criterio 1: media max_x en ({gs - 100000}, {gs}] n={len(w)}: {m:.0f} (umbral >= {GATE_MEAN}) -> {ok_m}")
    print(f"Criterio 2: det_max_x={d:.0f} (umbral > {GATE_DET}) -> {ok_d}")
    print("VEREDICTO:", "CONTINUAR" if (ok_m or ok_d) else "INTERRUMPIR y pasar al plan B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--window", type=int, default=100000)
    ap.add_argument("--gate-steps", type=int, default=GATE_STEPS)
    a = ap.parse_args()
    d = f"runs/{a.name}"
    prog, ep, ev = read(f"{d}/progress.csv"), read(f"{d}/episodes.csv"), read(f"{d}/eval.csv")
    if prog is None or ep is None or not len(prog):
        print("Faltan datos en", d)
        return
    last = prog.iloc[-1]
    steps = int(last["timesteps"])
    print(f"{a.name}: {steps} pasos | elapsed={last['elapsed_s'] / 60:.1f} min | fps={last['fps']:.0f} | episodios={len(ep)}")
    w = ep[ep["timesteps"] > steps - a.window]
    if len(w):
        mx = w["max_x"]
        print(f"Ultimos {a.window} pasos: n={len(w)} | max_x media={mx.mean():.0f} mediana={mx.median():.0f} p90={mx.quantile(0.9):.0f} max={mx.max():.0f}")
        print("  causas:", w["cause"].value_counts().to_dict(), "| banderas:", int(w["flag_get"].sum()))
    b = baseline_mean()
    if b is not None:
        print(f"  azar (50 episodios): media={b:.0f}")
    if ev is not None and len(ev):
        print("Evaluacion determinista (ultimas 6):")
        print(ev.tail(6).to_string(index=False))
    if steps >= a.gate_steps:
        gate(ep, ev, a.gate_steps)
    else:
        print(f"Puerta aun no evaluable: faltan {a.gate_steps - steps} pasos")


if __name__ == "__main__":
    main()
