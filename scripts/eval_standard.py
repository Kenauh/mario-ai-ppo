"""Evaluacion estandarizada (Fase 7): modelo SB3 sobre el entorno de entrenamiento.
mode stoch: muestrea, semilla = indice del episodio | mode det: argmax.
--noop-max N: antes de jugar, k decisiones NOOP con k = semilla % (N+1) (cada decision = 4 fotogramas)."""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
X_FLAG, SKIP = 3161, 4
FIELDS = ["label", "ckpt", "mode", "seed", "k_noop", "max_x", "norm", "flag", "cause", "decisions", "frames", "frames_info"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run_episode(env, model, mode, seed, noop_max):
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    obs, info = env.reset()
    k = seed % (noop_max + 1) if noop_max > 0 else 0
    term = trunc = False
    n_dec, max_x = 0, 0
    for _ in range(k):
        obs, _, term, trunc, info = env.step(0)
        n_dec += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
        if term or trunc:
            break
    while not (term or trunc):
        o = obs if obs.shape == model.observation_space.shape else np.transpose(obs, (2, 0, 1))
        act, _ = model.predict(o, deterministic=(mode == "det"))
        obs, _, term, trunc, info = env.step(int(act))
        n_dec += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
    flag = bool(info.get("flag_get", False))
    if flag:
        cause = "bandera"
    elif trunc:
        cause = "limite_fotogramas"
    elif int(info.get("time", 1)) == 0:
        cause = "reloj"
    else:
        cause = "muerte"
    return dict(seed=seed, k_noop=k, max_x=max_x, norm=round(min(1.0, max_x / X_FLAG), 4), flag=int(flag),
                cause=cause, decisions=n_dec, frames=n_dec * SKIP, frames_info=info.get("frames", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=["stoch", "det"], required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--noop-max", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=8000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from mario_agent.env_factory import make_mario_env
    model = PPO.load(a.ckpt, device="cpu")
    env = make_mario_env(max_frames=a.max_frames)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    for seed in range(a.episodes):
        r = run_episode(env, model, a.mode, seed, a.noop_max)
        r.update(label=a.label, ckpt=Path(a.ckpt).name, mode=a.mode)
        rows.append(r)
    env.close()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n, ok = len(rows), [r for r in rows if r["flag"]]
    lo, hi = wilson(len(ok), n)
    mx = np.array([r["max_x"] for r in rows])
    fr = np.array([r["frames"] for r in ok])
    print(f"[{a.label} {a.mode} noop<={a.noop_max}] n={n} banderas={len(ok)} tasa={len(ok) / n:.3f} IC95=[{lo:.3f}, {hi:.3f}]")
    exitos = f"media={fr.mean():.0f} mediana={np.median(fr):.0f} min={fr.min()}" if len(fr) else "sin exitos"
    print(f"  progreso norm media={mx.mean() / X_FLAG:.3f} | max_x mediana={np.median(mx):.0f} | frames exito: {exitos}")
    print(f"  causas={dict(Counter(r['cause'] for r in rows))} | {time.time() - t0:.0f} s | {a.out}")


if __name__ == "__main__":
    main()
