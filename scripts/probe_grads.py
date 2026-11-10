"""Sonda de gradientes (diagnostico, no entrena): ¿domina el critico sobre la politica?

  python scripts/probe_grads.py --ckpt models/speedtest/ckpt_80896_steps.zip
Aproximacion: retornos de Monte Carlo (no GAE) y ratio=1 (primera epoca de PPO).
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from stable_baselines3 import PPO

from mario_agent.env_factory import make_mario_env


def collect(model, n_min, gamma, seed):
    env = make_mario_env()
    torch.manual_seed(seed)
    O, A, G, n_eps = [], [], [], 0
    while len(O) < n_min:
        obs, _ = env.reset(seed=0)
        eo, ea, er = [], [], []
        while True:
            x = torch.as_tensor(obs[None]).permute(0, 3, 1, 2).contiguous()
            with torch.no_grad():
                p = torch.softmax(model.policy.get_distribution(x).distribution.logits, dim=1)[0]
            a = int(torch.multinomial(p, 1))
            eo.append(obs)
            ea.append(a)
            obs, r, term, trunc, _ = env.step(a)
            er.append(r)
            if term or trunc:
                break
        g, ret = 0.0, []
        for r in reversed(er):          # episodio truncado: sin bootstrap (aproximacion)
            g = r + gamma * g
            ret.append(g)
        O += eo
        A += ea
        G += ret[::-1]
        n_eps += 1
    env.close()
    return np.stack(O), np.array(A), np.array(G, dtype=np.float32), n_eps


def gnorm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-scale", type=float, default=1.0)
    args = ap.parse_args()

    model = PPO.load(args.ckpt, device="cpu")
    pol = model.policy
    O, A, G, n_eps = collect(model, args.n, model.gamma, args.seed)
    G = G * np.float32(args.reward_scale)   # el critico se entreno con la recompensa escalada
    obs_t = torch.as_tensor(O).permute(0, 3, 1, 2).contiguous()
    values, logp, ent = pol.evaluate_actions(obs_t, torch.as_tensor(A))
    values = values.flatten()
    ret_t = torch.as_tensor(G)
    adv = (ret_t - values).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    l_pg = -(adv * logp).mean()
    l_vf = ((ret_t - values) ** 2).mean()
    l_ent = -ent.mean()
    total = l_pg + model.ent_coef * l_ent + model.vf_coef * l_vf

    params = [p for p in pol.parameters() if p.requires_grad]
    shared = [p for n, p in pol.named_parameters() if n.startswith("features_extractor")]
    v = values.detach().numpy()
    ev = 1 - np.var(G - v) / np.var(G)
    print(f"decisiones={len(A)} episodios={n_eps} | retorno: media={G.mean():.1f} desv={G.std():.1f} | "
          f"MSE critico={l_vf.item():.1f} | var. explicada (MC)={ev:.2f} | entropia media={-l_ent.item():.3f}")
    print(f"{'norma del gradiente':<30}{'toda la red':>12}{'extractor':>11}")
    rows = [("politica (adv. normalizada)", l_pg),
            (f"critico x vf_coef={model.vf_coef}", model.vf_coef * l_vf),
            (f"entropia x ent_coef={model.ent_coef}", model.ent_coef * l_ent),
            ("suma (lo que se optimiza)", total)]
    for name, l in rows:
        print(f"{name:<30}{gnorm(l, params):>12.4f}{gnorm(l, shared):>11.4f}")
    tn = gnorm(total, params)
    print(f"max_grad_norm={model.max_grad_norm} -> factor de recorte = {min(1.0, model.max_grad_norm / tn):.4f}")


if __name__ == "__main__":
    main()
