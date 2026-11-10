"""Prueba que OpenCV funciona con numpy 1.x sobre una imagen real del juego."""
import cv2
import numpy as np
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = JoypadSpace(SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1)), SIMPLE_MOVEMENT)
obs = env.reset()
gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
print("gray:", gray.shape, gray.dtype, "| small:", small.shape, small.dtype, "| min/max:", small.min(), small.max())
assert isinstance(small, np.ndarray) and small.shape == (84, 84)
env.close()
print("OK: OpenCV funciona")
