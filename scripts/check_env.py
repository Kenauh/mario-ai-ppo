"""Verifica que el entorno de Mario carga y avanza unos pasos."""
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Se crea el entorno directo (sin gym.make) para evitar el TimeLimit de gym 0.26,
# que es incompatible con la API antigua de nes-py.
env = SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(1, 1))
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Acciones disponibles:", len(SIMPLE_MOVEMENT))
print("Espacio de observacion:", env.observation_space)
print("Espacio de acciones:", env.action_space)

obs = env.reset()
print("Tipo de retorno de reset():", type(obs), obs.shape)

steps = 0
total_reward = 0.0
for _ in range(300):
    obs, reward, done, info = env.step(env.action_space.sample())
    total_reward += reward
    steps += 1
    if done:
        break

print("Pasos ejecutados:", steps)
print("Recompensa acumulada:", total_reward)
print("Forma de la observacion:", obs.shape)
print("Claves de info:", sorted(info.keys()))
print("Info final:", info)
env.close()
print("OK: el entorno corre")
