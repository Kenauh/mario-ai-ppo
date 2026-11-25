"""Reproduce una grabacion de fotogramas crudos con el agente portable y genera un video.
Uso: python scripts/replay_video.py MODELO.pt frames_full.npz [salida.mp4]
Los fotogramas vienen de la grabacion (el emulador corrio en otra maquina); las acciones las calcula AQUI el agente."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mario_agent.agent import MarioAgent

model, fx = Path(sys.argv[1]), Path(sys.argv[2])
out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("replay_laptop.mp4")
d = np.load(fx)
frames, rec = d["frames"], d["actions"]
xs = d["x"] if "x" in d.files else None
agent = MarioAgent(str(model), device="cpu", deterministic=True, seed=0)
agent.reset()
h, w = frames.shape[1:3]
scale = 2
size = (w * scale, h * scale)
wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 60, size)
if not wr.isOpened():
    out = out.with_suffix(".avi")
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), 60, size)
bad, t0 = 0, time.time()
for i, f in enumerate(frames):
    a = agent.act(f)
    same = a == int(rec[i])
    bad += 0 if same else 1
    img = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
    x = "" if xs is None else f"  x={int(xs[i - 1] if i else xs[0])}"
    lines = [f"fotograma={i + 1}{x}", f"accion (calculada en esta maquina)={'+'.join(agent.buttons[a])}",
             f"igual a la grabada: {'si' if same else 'NO'}"]
    y = img.shape[0] - 8 - 18 * (len(lines) - 1)
    for t in lines:
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 18
    wr.write(img)
wr.release()
agent.close()
print(f"{len(frames)} fotogramas | acciones distintas = {bad} | {time.time() - t0:.1f} s | video: {out}")
print("OK: el agente de esta maquina reproduce toda la partida grabada" if bad == 0 else "FALLA: hay acciones distintas")
