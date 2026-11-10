"""Diagnostico: politica SB3 vs agente exportado, determinista vs estocastico.

  python scripts/diagnose.py --name speedtest --episodes 5
"""
import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from stable_baselines3 import PPO

from mario_agent.actions import SIMPLE_MOVEMENT_BUTTONS as BUTTONS
from mario_agent.agent import MarioAgent
from mario_agent.env_factory import make_mario_env
from mario_agent.export import export_policy
from mario_agent.wrappers import LegacyNESToGymnasium

NAMES = ["+".join(b) for b in BUTTONS]


def latest_ckpt(d):
    f = sorted(d.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    return f[-1] if f else None


def sb3_probs(model, obs_hwc):
    x = torch.as_tensor(obs_hwc[None]).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        logits = model.policy.get_distribution(x).distribution.logits
    return torch.softmax(logits, dim=1)[0]


def start_state(model, agent):
    env = make_mario_env()
    obs, _ = env.reset(seed=0)
    p = sb3_probs(model, obs)
    with torch.no_grad():
        x = torch.as_tensor(obs[None]).permute(0, 3, 1, 2).contiguous()
        q = torch.softmax(agent.net(x), dim=1)[0]
    ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum())
    print("\n== Estado inicial (reset) ==")
    print(f"{'accion':<14}{'P(SB3)':>9}{'P(agente)':>11}")
    for i, n in enumerate(NAMES):
        print(f"{n:<14}{p[i].item():>9.3f}{q[i].item():>11.3f}")
    print(f"entropia SB3 = {ent:.3f} nats (maximo con 7 acciones = 1.946) | "
          f"max|P_sb3 - P_agente| = {(p - q).abs().max().item():.2e}")
    print(f"argmax SB3 = {NAMES[int(p.argmax())]} | argmax agente = {NAMES[int(q.argmax())]}")
    env.close()


def pick(p, deterministic):
    return int(p.argmax()) if deterministic else int(torch.multinomial(p, 1))


def episode_sb3(model, deterministic, seed, max_frames):
    env = make_mario_env(max_frames=max_frames)
    obs, _ = env.reset(seed=0)
    torch.manual_seed(seed)
    acts, max_x, info = [], 0, {}
    while True:
        a = pick(sb3_probs(model, obs), deterministic)
        acts.append(a)
        obs, _, term, trunc, info = env.step(a)
        max_x = max(max_x, int(info.get("x_pos", 0)))
        if term or trunc:
            break
    env.close()
    cause = "bandera" if info.get("flag_get") else ("limite" if trunc and not term else "muerte")
    return dict(max_x=max_x, frames=info["frames"], cause=cause, acts=acts)


def episode_agent(path, deterministic, seed, max_frames):
    agent = MarioAgent(path, device="cpu", deterministic=deterministic, seed=seed)
    env = LegacyNESToGymnasium()
    obs, _ = env.reset(seed=0)
    agent.reset()
    acts, max_x, frames, info, done = [], 0, 0, {}, False
    while frames < max_frames and not done:
        a = agent.act(obs)
        if frames % agent.decide_every == 0:
            acts.append(a)
        obs, _, done, _, info = env.step(a)
        frames += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
    env.close()
    cause = "bandera" if info.get("flag_get") else ("muerte" if done else "limite")
    return dict(max_x=max_x, frames=frames, cause=cause, acts=acts)


def summarize(label, runs):
    mx = [r["max_x"] for r in runs]
    causes = dict(Counter(r["cause"] for r in runs))
    allacts = Counter(a for r in runs for a in r["acts"])
    tot = max(sum(allacts.values()), 1)
    top = ", ".join(f"{NAMES[a]} {100 * n / tot:.0f}%" for a, n in allacts.most_common(3))
    print(f"{label:<17}n={len(runs)} max_x: media={np.mean(mx):.0f} min={min(mx)} max={max(mx)} "
          f"| causas={causes} | acciones: {top}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="speedtest")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=3000)
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(ROOT / "models" / args.name)
    if ckpt is None or not ckpt.exists():
        sys.exit("No encontre checkpoint.")
    model = PPO.load(ckpt, device="cpu")
    steps = int(model.num_timesteps)
    pt = export_policy(model, ROOT / "models" / args.name / "export" / f"diag_{steps}.pt")
    agent = MarioAgent(pt, device="cpu")
    print(f"Checkpoint: {ckpt.name} ({steps} pasos)")
    start_state(model, agent)

    mf, n = args.max_frames, args.episodes
    print(f"\n== Episodios desde el reset (max_frames={mf}) ==")
    d = summarize("SB3 determ.", [episode_sb3(model, True, s, mf) for s in range(2)])
    summarize("Agente determ.", [episode_agent(pt, True, s, mf) for s in range(2)])
    summarize("SB3 estocast.", [episode_sb3(model, False, s, mf) for s in range(n)])
    summarize("Agente estocast.", [episode_agent(pt, False, s, mf) for s in range(n)])
    print("Determinista reproducible (2 corridas iguales):", d[0]["acts"] == d[1]["acts"])
    print("Primeras 20 decisiones SB3 determinista:", [NAMES[a] for a in d[0]["acts"][:20]])


if __name__ == "__main__":
    main()
