"""Prueba que el agente exportado (PyTorch puro) reproduce a SB3 exactamente.

A) Logits identicos sobre observaciones reales del entorno.
B) Con fotogramas crudos RGB (uno por llamada, como lo daria un evaluador):
   mismo apilado que el entorno de entrenamiento y misma accion que SB3.
"""
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mario_agent.agent import MarioAgent
from mario_agent.env_factory import make_mario_env
from mario_agent.export import export_policy
from mario_agent.wrappers import LegacyNESToGymnasium


def main():
    tmp = Path(tempfile.mkdtemp())
    model = PPO("CnnPolicy", DummyVecEnv([make_mario_env]), device="cpu", seed=0, verbose=0)
    torch.manual_seed(0)
    for m in model.policy.features_extractor.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight)
    torch.nn.init.normal_(model.policy.action_net.weight, std=1.0)  # acciones distintas entre observaciones

    pt = export_policy(model, tmp / "policy.pt")
    print("Exportado:", pt, f"({pt.stat().st_size / 1e6:.2f} MB)")
    agent = MarioAgent(pt, device="cpu", deterministic=True)

    def sb3_logits(batch_hwc):
        x = torch.as_tensor(batch_hwc).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            return torch.log_softmax(model.policy.get_distribution(x).distribution.logits, dim=1)

    # ---- A) logits sobre observaciones reales ----
    env = make_mario_env()
    obs, _ = env.reset(seed=0)
    batch = [obs]
    for _ in range(63):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
        batch.append(obs)
    batch = np.stack(batch)
    ref = sb3_logits(batch)
    with torch.no_grad():
        mine = torch.log_softmax(agent.net(torch.as_tensor(batch).permute(0, 3, 1, 2).contiguous()), dim=1)
    diff = (ref - mine).abs().max().item()
    same = bool((ref.argmax(1) == mine.argmax(1)).all())
    print(f"A) max|logits SB3 - agente| = {diff:.2e} | mismas acciones en 64 obs: {same} "
          f"| acciones distintas usadas: {len(set(ref.argmax(1).tolist()))}")
    assert diff < 1e-3 and same, "FALLO A: el agente exportado NO reproduce a SB3"
    env.close()

    # ---- B) fotogramas crudos, un fotograma por llamada ----
    env_w, env_r = make_mario_env(), LegacyNESToGymnasium()
    n_decisions, lat = 0, []
    for ep in range(3):
        obs_w, _ = env_w.reset(seed=ep)
        obs_r, _ = env_r.reset(seed=ep)
        agent.reset()
        for _ in range(400):
            a_ref = int(sb3_logits(obs_w[None]).argmax(1).item())
            t0 = time.perf_counter()
            a_ag = agent.act(obs_r)                       # llamada de decision
            lat.append(time.perf_counter() - t0)
            assert np.array_equal(agent.current_stack(), obs_w), "FALLO B: apilado distinto al del entorno"
            assert a_ag == a_ref, f"FALLO B: accion {a_ag} != SB3 {a_ref}"
            n_decisions += 1

            done_r = False
            for k in range(4):                            # 4 fotogramas crudos con la misma accion
                obs_r, _, done_r, _, _ = env_r.step(a_ref)
                if done_r:
                    break
                if k < 3:
                    assert agent.act(obs_r) == a_ref, "FALLO B: no repitio la accion entre decisiones"
            obs_w, _, term, trunc, _ = env_w.step(a_ref)
            if done_r or term or trunc:
                break
    print(f"B) {n_decisions} decisiones en 3 episodios: apilado identico y accion identica a SB3")
    print(f"   latencia por decision (CPU): media {1000 * np.mean(lat):.2f} ms, max {1000 * np.max(lat):.2f} ms")
    env_w.close(); env_r.close(); agent.close()
    print("OK: paridad verificada")


if __name__ == "__main__":
    main()
