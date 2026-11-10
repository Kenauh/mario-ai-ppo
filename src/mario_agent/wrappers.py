"""Wrappers propios: puente gym->gymnasium y preprocesamiento de observaciones.

Los mismos wrappers se usan en entrenamiento y en el agente entregable, para que
el preprocesamiento sea idéntico en ambos.
"""
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from .preprocess import preprocess_frame


class LegacyNESToGymnasium(gym.Env):
    """Puente: adapta el entorno antiguo (API gym/nes-py) a la API de gymnasium.

    Antigua: reset() -> obs ; step() -> (obs, reward, done, info)
    Nueva:   reset() -> (obs, info) ; step() -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self, actions=SIMPLE_MOVEMENT, rom_mode="vanilla", target=(1, 1)):
        super().__init__()
        # Se crea directo (sin gym.make) para evitar el TimeLimit de gym 0.26.
        raw = SuperMarioBrosEnv(rom_mode=rom_mode, lost_levels=False, target=target)
        self._env = JoypadSpace(raw, actions)
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(0, 255, (240, 256, 3), np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return np.array(obs, copy=True), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        # copy: por seguridad, para no depender de un buffer interno reutilizado.
        # 'done' del juego = fin de episodio natural (muerte o bandera) -> terminated.
        return np.array(obs, copy=True), float(reward), bool(done), False, info

    def close(self):
        self._env.close()


class SkipFrame(gym.Wrapper):
    """Repite la acción 'skip' fotogramas y suma recompensas. Cuenta fotogramas reales."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self._emu_frames = 0

    def reset(self, **kwargs):
        self._emu_frames = 0
        obs, info = self.env.reset(**kwargs)
        info["frames"] = 0
        return obs, info

    def step(self, action):
        total = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            self._emu_frames += 1
            if terminated or truncated:
                break
        info["frames"] = self._emu_frames
        return obs, total, terminated, truncated, info


class MaxFrames(gym.Wrapper):
    """Trunca el episodio al llegar a 'max_frames' fotogramas (PROVISIONAL: lo fija el anexo)."""

    def __init__(self, env, max_frames=8000):
        super().__init__(env)
        self.max_frames = max_frames

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated and info.get("frames", 0) >= self.max_frames:
            truncated = True
        return obs, reward, terminated, truncated, info


class GrayResize(gym.ObservationWrapper):
    """Escala de grises + reducción a size x size. Salida: (size, size, 1) uint8."""

    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(0, 255, (size, size, 1), np.uint8)

    def observation(self, obs):
        return preprocess_frame(obs, self.size)[:, :, None]


class FrameStack(gym.Wrapper):
    """Apila los últimos n fotogramas en el último eje. Salida: (H, W, n) uint8.

    Al reiniciar se rellena repitiendo el primer fotograma.
    """

    def __init__(self, env, n=4):
        super().__init__(env)
        self.n = n
        self._buffer = deque(maxlen=n)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(0, 255, (h, w, c * n), np.uint8)

    def _stacked(self):
        return np.concatenate(list(self._buffer), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n):
            self._buffer.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs)
        return self._stacked(), reward, terminated, truncated, info


class RewardScale(gym.RewardWrapper):
    """Multiplica la recompensa por una constante. SOLO entrenamiento (el agente no usa la recompensa)."""

    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return reward * self.scale
