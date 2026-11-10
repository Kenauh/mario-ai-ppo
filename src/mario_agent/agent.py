"""Agente entregable. Solo depende de PyTorch, NumPy y OpenCV.

Interfaz:  MarioAgent(model_path) -> reset() -> act(obs) -> close()
  obs: imagen RGB (240, 256, 3) uint8 (un fotograma del juego).
  act devuelve el indice de la accion (int). act_buttons devuelve los botones.

Se mantiene el mismo comportamiento que en entrenamiento: se decide cada
'decide_every' llamadas (salto de fotogramas) y se repite la ultima accion entre medias.
La politica esta congelada: no se actualiza nunca durante la ejecucion.
"""
from collections import deque

import numpy as np
import torch

from .policy import load_policy
from .preprocess import preprocess_frame


class MarioAgent:
    def __init__(self, model_path, device="cpu", deterministic=True, decide_every=None,
                 color_order=None, seed=0):
        self.net, self.meta = load_policy(model_path, device)
        self.device = device
        self.deterministic = deterministic
        self.decide_every = int(decide_every if decide_every is not None else self.meta["skip"])
        self.color_order = color_order or self.meta["color_order"]
        self.buttons = self.meta["buttons"]
        self._n = self.meta["in_channels"]
        self._size = self.meta["size"]
        self._seed = seed
        self._gen = torch.Generator(device="cpu")
        self.reset()

    def reset(self):
        """Reinicia el estado interno (buffer de fotogramas, contador y semilla)."""
        self._frames = deque(maxlen=self._n)
        self._calls = 0
        self._last_action = 0
        self._gen.manual_seed(self._seed)

    def act(self, observation) -> int:
        if self._calls % self.decide_every == 0:
            self._last_action = self._decide(observation)
        self._calls += 1
        return self._last_action

    def act_buttons(self, observation):
        return self.buttons[self.act(observation)]

    def current_stack(self) -> np.ndarray:
        """(H, W, n) uint8: el mismo formato que devuelve el entorno de entrenamiento."""
        return np.stack(list(self._frames), axis=-1)

    def close(self):
        self._frames.clear()

    # ---- internos ----
    def _prepare(self, observation) -> np.ndarray:
        obs = np.asarray(observation)
        if obs.ndim != 3 or obs.shape[2] != 3 or obs.dtype != np.uint8:
            raise ValueError(f"Se esperaba imagen RGB uint8 (H, W, 3); llego {obs.shape} {obs.dtype}")
        if self.color_order == "BGR":
            obs = obs[:, :, ::-1]
        return preprocess_frame(obs, self._size)

    def _decide(self, observation) -> int:
        frame = self._prepare(observation)
        if not self._frames:                      # primer fotograma: se repite (igual que en entrenamiento)
            for _ in range(self._n):
                self._frames.append(frame)
        else:
            self._frames.append(frame)
        x = torch.from_numpy(np.stack(list(self._frames), axis=0)[None]).to(self.device)
        with torch.inference_mode():
            logits = self.net(x)
        if self.deterministic:
            return int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits.cpu(), dim=1)
        return int(torch.multinomial(probs, 1, generator=self._gen).item())
