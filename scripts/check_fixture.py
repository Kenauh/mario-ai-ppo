"""Reproduce la secuencia grabada. Uso: python scripts/check_fixture.py MODELO.pt frames_actions.npz
Solo necesita torch, numpy y opencv (sin emulador, SB3 ni Gym)."""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
model, fx = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(fx)
frames, acts = d["frames"], d["actions"]
print("sha256 del modelo igual al de la grabacion:", hashlib.sha256(model.read_bytes()).hexdigest() == str(d["model_sha256"]))
import cv2
import torch
print("versiones: python", sys.version.split()[0], "| torch", torch.__version__, "| numpy", np.__version__, "| opencv", cv2.__version__)
from mario_agent.agent import MarioAgent

agent = MarioAgent(str(model), device="cpu", deterministic=True, seed=0)
agent.reset()
t0 = time.time()
out = [agent.act(f) for f in frames]
bad = int((np.array(out) != acts).sum())
print(f"{len(frames)} fotogramas: acciones distintas = {bad} | {1000 * (time.time() - t0) / len(frames):.2f} ms por llamada")
print("OK: mismas acciones" if bad == 0 else "FALLA: hay acciones distintas")
