"""Juega una partida EN VIVO con el agente portable (.pt) sobre el emulador y graba un video.
Solo necesita torch, numpy, opencv y el emulador (nes-py + gym-super-mario-bros); NO usa SB3.
Uso: python scripts/play.py MODELO.pt [--stochastic] [--seed N] [--noop-k K] [--max-frames 8000] [--out video.mp4]"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mario_agent.agent import MarioAgent
from mario_agent.wrappers import LegacyNESToGymnasium

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--stochastic", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--noop-k", type=int, default=0, help="decisiones NOOP iniciales (4 fotogramas cada una)")
ap.add_argument("--max-frames", type=int, default=8000)
ap.add_argument("--scale", type=int, default=2)
ap.add_argument("--out", default="partida.mp4")
a = ap.parse_args()


def draw(frame_rgb, scale, lines):
    img = cv2.resize(frame_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    y = img.shape[0] - 8 - 18 * (len(lines) - 1)
    for t in lines:
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 18
    return img


agent = MarioAgent(a.model, device="cpu", deterministic=not a.stochastic, seed=a.seed)
env = LegacyNESToGymnasium()
obs, _ = env.reset(seed=a.seed)
agent.reset()
h, w = obs.shape[:2]
out = Path(a.out)
wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 60, (w * a.scale, h * a.scale))
if not wr.isOpened():
    out = out.with_suffix(".avi")
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), 60, (w * a.scale, h * a.scale))
frames, max_x, cause, info, t0 = 0, 0, "limite_fotogramas", {}, time.time()
while frames < a.max_frames:
    action = agent.act(obs)
    if frames < 4 * a.noop_k:
        action = 0
    obs, _, done, _, info = env.step(action)
    frames += 1
    max_x = max(max_x, int(info.get("x_pos", 0)))
    wr.write(draw(obs, a.scale, [f"x={info.get('x_pos')}  fotograma={frames}  tiempo={info.get('time')}",
                                 f"accion={'+'.join(agent.buttons[action])}"]))
    if done:
        cause = "bandera" if info.get("flag_get") else "muerte"
        break
wr.release()
agent.close()
env.close()
print(f"Resultado: causa={cause} | max_x={max_x} | fotogramas={frames} | flag_get={bool(info.get('flag_get'))} | {time.time() - t0:.1f} s | video: {out}")
