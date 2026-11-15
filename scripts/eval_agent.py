"""Evalua el AGENTE PORTABLE (PyTorch/NumPy/OpenCV, sin SB3) sobre fotogramas crudos.
Misma convencion que eval_standard.py: k = semilla % (noop-max+1) decisiones NOOP (4 fotogramas c/u) antes de actuar."""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
FIELDS = ["label", "mode", "seed", "k_noop", "max_x", "flag", "cause", "frames"]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run_episode(env, model_path, mode, seed, noop_max, max_frames):
    from mario_agent.agent import MarioAgent
    agent = MarioAgent(model_path, device="cpu", deterministic=(mode == "det"), seed=seed)
    obs, _ = env.reset(seed=seed)
    agent.reset()
    k = seed % (noop_max + 1) if noop_max > 0 else 0
    frames, max_x, flag, cause = 0, 0, False, "limite_fotogramas"
    while frames < max_frames:
        action = agent.act(obs)
        if frames < 4 * k:
            action = 0
        obs, _, done, _, info = env.step(action)
        frames += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
        if done:
            flag = bool(info.get("flag_get"))
            cause = "bandera" if flag else "muerte"
            break
    agent.close()
    return dict(seed=seed, k_noop=k, max_x=max_x, flag=int(flag), cause=cause, frames=frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=["stoch", "det"], required=True)
    ap.add_argument("--episodes", type=int, default=31)
    ap.add_argument("--noop-max", type=int, default=0)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=8000)
    ap.add_argument("--compare", default=None, help="CSV de eval_standard.py con las mismas semillas (modo det)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from mario_agent.wrappers import LegacyNESToGymnasium
    env = LegacyNESToGymnasium()
    t0, rows = time.time(), []
    for seed in range(a.seed_start, a.seed_start + a.episodes):
        r = run_episode(env, a.model, a.mode, seed, a.noop_max, a.max_frames)
        r.update(label=a.label, mode=a.mode)
        rows.append(r)
    env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n, ok = len(rows), [r for r in rows if r["flag"]]
    lo, hi = wilson(len(ok), n)
    fr = np.array([r["frames"] for r in ok])
    exitos = f"mediana={np.median(fr):.0f} min={fr.min()}" if len(fr) else "sin exitos"
    print(f"[agente portable {a.label} {a.mode}] n={n} banderas={len(ok)} tasa={len(ok) / n:.3f} IC95=[{lo:.3f}, {hi:.3f}] | frames exito: {exitos}")
    print(f"  causas={dict(Counter(r['cause'] for r in rows))} | {time.time() - t0:.0f} s | {a.out}")
    if a.compare:
        import pandas as pd
        ref = pd.read_csv(a.compare).set_index("seed")
        j = pd.DataFrame(rows).set_index("seed").join(ref[["flag", "frames_info", "max_x"]], rsuffix="_sb3", how="inner")
        print(f"  vs SB3 ({len(j)} semillas): misma bandera {int((j.flag == j.flag_sb3).sum())}, "
              f"mismo max_x {int((j.max_x == j.max_x_sb3).sum())}, mismos fotogramas {int((j.frames == j.frames_info).sum())}")
        bad = j[(j.flag != j.flag_sb3) | (j.frames != j.frames_info) | (j.max_x != j.max_x_sb3)]
        if len(bad):
            print(bad[["k_noop", "flag", "flag_sb3", "max_x", "max_x_sb3", "frames", "frames_info"]].to_string())


if __name__ == "__main__":
    main()
