"""Fábrica del entorno: ensambla el puente y los wrappers en el orden correcto."""
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from .wrappers import FrameStack, GrayResize, LegacyNESToGymnasium, MaxFrames, SkipFrame


def make_mario_env(skip=4, max_frames=8000, size=84, n_stack=4, actions=SIMPLE_MOVEMENT):
    """Entorno final: observación (size, size, n_stack) uint8, acciones Discrete(len(actions)).

    El orden importa: MaxFrames va después de SkipFrame porque usa info['frames'].
    """
    env = LegacyNESToGymnasium(actions)
    env = SkipFrame(env, skip)
    env = MaxFrames(env, max_frames)
    env = GrayResize(env, size)
    env = FrameStack(env, n_stack)
    return env
