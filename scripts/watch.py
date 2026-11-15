"""Graba un episodio del agente: checkpoint SB3 -> .pt -> MarioAgent -> video.

El agente recibe fotogramas crudos RGB, uno por llamada, como lo haria un evaluador.

  python scripts/watch.py --name speedtest
  python scripts/watch.py --name cfg1_base --ckpt models/cfg1_base/ckpt_500096_steps.zip
  python scripts/watch.py --name cfg1_base --stochastic --seed 3
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from mario_agent.agent import MarioAgent
from mario_agent.export import export_policy
from mario_agent.wrappers import LegacyNESToGymnasium


def latest_ckpt(models_dir):
    files = sorted(models_dir.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    return files[-1] if files else None


def open_writer(path, size, fps=60):
    """mp4 si el codec esta disponible; si no, avi MJPG."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if w.isOpened():
        return w, path
    path = path.with_suffix(".avi")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not w.isOpened():
        sys.exit("No pude abrir un escritor de video (mp4v ni MJPG).")
    return w, path


def draw(frame_rgb, scale, lines):
    img = cv2.resize(frame_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    y = img.shape[0] - 8 - 18 * (len(lines) - 1)
    for t in lines:
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 18
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cfg1_base")
    ap.add_argument("--ckpt", default=None, help="ruta a un .zip; por defecto el ultimo de models/<name>")
    ap.add_argument("--stochastic", action="store_true", help="muestrea acciones en vez de la mas probable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=8000)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--noop-k", type=int, default=0, help="decisiones NOOP iniciales (4 fotogramas cada una)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(ROOT / "models" / args.name)
    if ckpt is None or not ckpt.exists():
        sys.exit("No encontre checkpoint. Usa --ckpt o revisa models/" + args.name)

    from stable_baselines3 import PPO   # solo para convertir el checkpoint; el agente no lo usa
    model = PPO.load(ckpt, device="cpu")
    steps = int(model.num_timesteps)
    pt = export_policy(model, ROOT / "models" / args.name / "export" / f"policy_{steps}.pt")
    del model
    agent = MarioAgent(pt, device="cpu", deterministic=not args.stochastic, seed=args.seed)
    modo = "estocastico" if args.stochastic else "determinista"
    print(f"Checkpoint: {ckpt.name} ({steps} pasos) -> {pt.name} | modo: {modo}")

    env = LegacyNESToGymnasium()
    obs, _ = env.reset(seed=args.seed)
    agent.reset()
    out_dir = ROOT / "results" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("stoch" if args.stochastic else "det") + (f"_k{args.noop_k}" if args.noop_k else "")
    h, w = obs.shape[:2]
    writer, vpath = open_writer(out_dir / f"{args.name}_{steps}_{tag}_s{args.seed}.mp4",
                                (w * args.scale, h * args.scale))

    frames, max_x, cause, info = 0, 0, "limite_fotogramas", {}
    t0 = time.time()
    while frames < args.max_frames:
        action = agent.act(obs)  # un fotograma crudo por llamada
        if frames < 4 * args.noop_k:
            action = 0  # espera inicial: NOOP; el agente igual recibe los fotogramas
        obs, _, done, _, info = env.step(action)
        frames += 1
        max_x = max(max_x, int(info.get("x_pos", 0)))
        label = "+".join(agent.buttons[action])
        l1 = f"x={info.get('x_pos')}  fotograma={frames}  tiempo={info.get('time')}"
        writer.write(draw(obs, args.scale, [l1, f"accion={label}"]))
        if done:
            cause = "bandera" if info.get("flag_get") else "muerte"
            break
    writer.release()
    agent.close()
    env.close()
    print(f"Video: {vpath}")
    print(f"Resultado: causa={cause} | max_x={max_x} | fotogramas={frames} | "
          f"flag_get={bool(info.get('flag_get'))} | {time.time() - t0:.1f}s de calculo")


if __name__ == "__main__":
    main()
