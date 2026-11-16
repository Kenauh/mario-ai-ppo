"""Graba fotogramas crudos y las acciones del agente (determinista) para probarlo en OTRA maquina sin emulador.
Uso: python scripts/make_fixture.py MODELO.pt [N=600] [SALIDA.npz]"""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
model = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "fixture" / "frames_actions.npz"
from mario_agent.agent import MarioAgent
from mario_agent.wrappers import LegacyNESToGymnasium

env = LegacyNESToGymnasium()
agent = MarioAgent(str(model), device="cpu", deterministic=True, seed=0)
obs, _ = env.reset(seed=0)
agent.reset()
frames, acts = [], []
for _ in range(N):
    a = agent.act(obs)
    frames.append(obs.copy())
    acts.append(a)
    obs, _, done, _, info = env.step(a)
    if done:
        break
env.close()
agent.close()
out.parent.mkdir(parents=True, exist_ok=True)
sha = hashlib.sha256(model.read_bytes()).hexdigest()
np.savez_compressed(out, frames=np.stack(frames), actions=np.array(acts, dtype=np.int16), model_sha256=sha)
print(len(frames), "fotogramas ->", out, f"{out.stat().st_size / 1e6:.1f} MB | sha256 del modelo:", sha[:16])
