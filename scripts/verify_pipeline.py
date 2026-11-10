"""Verifica puente + wrappers + compatibilidad con SB3 + entornos paralelos."""
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
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from mario_agent.env_factory import make_mario_env


def run_single(n_episodes=3):
    env = make_mario_env()
    env.action_space.seed(0)
    print("Observacion:", env.observation_space)
    print("Acciones:", env.action_space)

    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 4) and obs.dtype == np.uint8
    obs_b, _ = env.reset(seed=0)
    print("Reset reproducible (mismo frame inicial):", bool(np.array_equal(obs, obs_b)))

    # Imagen de muestra: 4 fotogramas apilados tras avanzar con derecha+A
    for _ in range(40):
        obs, *_ = env.step(2)
    strip = np.hstack([obs[:, :, i] for i in range(4)])
    strip = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    (ROOT / "results").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "results" / "obs_sample.png"), strip)
    print("Muestra guardada en results/obs_sample.png")

    total_steps, t0 = 0, time.time()
    for ep in range(n_episodes):
        env.reset()
        steps = 0
        while True:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            steps += 1
            if term or trunc:
                break
        total_steps += steps
        causa = "bandera" if info.get("flag_get") else ("limite de fotogramas" if trunc else "muerte/fin")
        print(f"ep {ep}: decisiones={steps} fotogramas={info['frames']} x_pos={info['x_pos']} "
              f"vidas={info['life']} tiempo_juego={info['time']} causa={causa}")
    dt = time.time() - t0
    print(f"1 entorno: {total_steps} decisiones en {dt:.2f}s -> {total_steps / dt:.0f} decisiones/s")
    env.close()


def run_check_env():
    try:
        check_env(make_mario_env(), warn=True)
        print("SB3 check_env: OK")
    except Exception as e:
        print("SB3 check_env: FALLO ->", repr(e))


def run_vec(n_envs, n_steps=300):
    venv = SubprocVecEnv([make_mario_env for _ in range(n_envs)])
    obs = venv.reset()
    print(f"VecEnv x{n_envs}: obs {obs.shape}")
    t0 = time.time()
    for _ in range(n_steps):
        actions = np.array([venv.action_space.sample() for _ in range(n_envs)])
        obs, rewards, dones, infos = venv.step(actions)
    dt = time.time() - t0
    print(f"{n_envs} entornos: {n_envs * n_steps} decisiones en {dt:.2f}s -> {n_envs * n_steps / dt:.0f} decisiones/s")
    venv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()
    run_single()
    run_check_env()
    run_vec(args.n_envs)
    print("OK: pipeline verificado")
