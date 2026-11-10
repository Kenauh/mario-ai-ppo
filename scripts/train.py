"""Entrenamiento PPO (SB3) del agente de Mario.

  python scripts/train.py --config configs/base.json
  python scripts/train.py --config configs/base.json --resume     (continua desde el ultimo checkpoint)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from functools import partial
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from mario_agent.callbacks import EpisodeLogger
from mario_agent.env_factory import make_mario_env


def make_train_env(**env_kw):
    """Entorno de entrenamiento = entorno del agente + Monitor (solo para registrar)."""
    return Monitor(make_mario_env(**env_kw))


def latest_ckpt(models_dir):
    files = sorted(models_dir.glob("ckpt_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
    return files[-1] if files else None


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.json"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.name:
        cfg["name"] = args.name
    if args.total_steps:
        cfg["total_timesteps"] = args.total_steps
    if args.n_envs:
        cfg["n_envs"] = args.n_envs
    name = cfg["name"]
    models_dir, run_dir = ROOT / "models" / name, ROOT / "runs" / name
    models_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt = latest_ckpt(models_dir)
    if ckpt and not args.resume:
        sys.exit(f"Ya hay checkpoints en {models_dir}. Usa --resume para continuar o --name para otra corrida.")
    if args.resume and not ckpt:
        sys.exit("No hay checkpoints para reanudar.")

    set_random_seed(cfg["seed"])
    venv = SubprocVecEnv([partial(make_train_env, **cfg["env"]) for _ in range(cfg["n_envs"])])
    venv.seed(cfg["seed"])
    tb_kw = dict(tensorboard_log=str(run_dir), device=args.device)
    if ckpt:
        model = PPO.load(ckpt, env=venv, **tb_kw)
        print("Reanudando desde", ckpt.name, "| pasos previos:", model.num_timesteps)
    else:
        model = PPO("CnnPolicy", venv, seed=cfg["seed"], verbose=1, **cfg["ppo"], **tb_kw)

    pkgs = ["torch", "stable-baselines3", "gymnasium", "gym", "nes-py", "gym-super-mario-bros",
            "numpy", "opencv-python-headless"]
    info = {
        "unix_time": time.time(), "cmd": sys.argv, "config": cfg,
        "resume_from": ckpt.name if ckpt else None, "device": str(model.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(), "python": sys.version, "git_commit": git_head(),
        "versions": {p: version(p) for p in pkgs},
    }
    (run_dir / f"run_info_{int(time.time())}.json").write_text(json.dumps(info, indent=2))
    print("Dispositivo:", model.device, "| GPU:", info["gpu"], "| envs:", cfg["n_envs"])

    remaining = cfg["total_timesteps"] - model.num_timesteps
    if remaining <= 0:
        sys.exit("Ya se alcanzo total_timesteps.")
    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(cfg["checkpoint_every"] // cfg["n_envs"], 1),
                           save_path=str(models_dir), name_prefix="ckpt"),
        EpisodeLogger(run_dir),
    ])
    t0 = time.time()
    try:
        model.learn(total_timesteps=remaining, callback=callbacks,
                    reset_num_timesteps=False, tb_log_name="tb")
        model.save(models_dir / f"ckpt_{model.num_timesteps}_steps")
        model.save(models_dir / "final")
    except KeyboardInterrupt:
        model.save(models_dir / f"ckpt_{model.num_timesteps}_steps")
        print("Interrumpido: checkpoint guardado en el paso", model.num_timesteps)
    finally:
        venv.close()
    print(f"Segmento terminado en {(time.time() - t0) / 3600:.3f} h | pasos totales: {model.num_timesteps}")


if __name__ == "__main__":
    main()
