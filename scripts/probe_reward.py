"""Sonda de recompensa. NO es un agente: acciones constantes para entender la recompensa."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from mario_agent.actions import SIMPLE_MOVEMENT_BUTTONS as BUTTONS
from mario_agent.env_factory import make_mario_env

NAMES = ["+".join(b) for b in BUTTONS]
GAMMA, MAX_FRAMES = 0.99, 3000


def jump_pattern(t):   # 4 decisiones right+A+B y 2 right+B (suelta A para poder saltar de nuevo)
    return 4 if (t % 6) < 4 else 3


def run(policy):
    env = make_mario_env(max_frames=MAX_FRAMES)
    env.reset(seed=0)
    t, total, disc, max_x, info = 0, 0.0, 0.0, 0, {}
    while True:
        _, r, term, trunc, info = env.step(policy(t))
        total += r
        disc += (GAMMA ** t) * r
        max_x = max(max_x, int(info.get("x_pos", 0)))
        t += 1
        if term or trunc:
            break
    env.close()
    cause = "bandera" if info.get("flag_get") else ("limite" if trunc and not term else "muerte")
    return t, total, disc, max_x, cause


print(f"{'comportamiento':<22}{'decis.':>7}{'R total':>10}{'G(0.99)':>10}{'R/decis.':>10}{'max_x':>7}  causa")
rows = [(f"constante {n}", (lambda t, a=i: a)) for i, n in enumerate(NAMES)]
rows.append(("patron salto (4+2)", jump_pattern))
for label, pol in rows:
    t, tot, disc, mx, cause = run(pol)
    print(f"{label:<22}{t:>7}{tot:>10.1f}{disc:>10.1f}{tot / t:>10.2f}{mx:>7}  {cause}")
