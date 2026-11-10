"""Linea base: politica uniforme sobre las 7 acciones, mismo entorno y limite de fotogramas."""
import argparse
import csv
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np

from mario_agent.env_factory import make_mario_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_mario_env()
    n_act = env.action_space.n
    rows, t0 = [], time.time()
    for ep in range(args.episodes):
        env.reset(seed=0)
        max_x, steps = 0, 0
        while True:
            _, _, term, trunc, info = env.step(int(rng.integers(n_act)))
            steps += 1
            max_x = max(max_x, int(info.get("x_pos", 0)))
            if term or trunc:
                break
        cause = "bandera" if info.get("flag_get") else ("limite" if trunc and not term else "muerte")
        rows.append((ep, max_x, steps, int(info["frames"]), cause))
    env.close()

    mx = np.array([r[1] for r in rows])
    print(f"{args.episodes} episodios aleatorios en {time.time() - t0:.0f}s")
    print(f"max_x: media={mx.mean():.0f} mediana={np.median(mx):.0f} p10={np.percentile(mx, 10):.0f} "
          f"p90={np.percentile(mx, 90):.0f} max={mx.max()} | causas={dict(Counter(r[4] for r in rows))}")
    bins = Counter((int(x) // 100) * 100 for x in mx)
    print("Donde termina (max_x por tramos de 100):", dict(sorted(bins.items())))
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "baseline_random.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "max_x", "decisiones", "fotogramas", "causa"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
