"""Exporta la política de un modelo PPO de SB3 a un .pt ligero. (Único módulo que usa SB3.)"""
from pathlib import Path

import torch

from .actions import SIMPLE_MOVEMENT_BUTTONS
from .policy import NatureCNNPolicy

_PREFIXES = {
    "features_extractor.cnn.": "cnn.",
    "features_extractor.linear.": "linear.",
    "action_net.": "action_net.",
}


def extract_actor_state(sb3_policy):
    """Pesos que intervienen en la acción (no exporta la cabeza de valor: no decide acciones)."""
    extra = sum(p.numel() for p in sb3_policy.mlp_extractor.policy_net.parameters())
    if extra != 0:
        raise RuntimeError(
            f"mlp_extractor.policy_net tiene {extra} parametros; esta exportacion asume net_arch vacia. "
            "Hay que extender policy.py antes de exportar."
        )
    state = {}
    for key, tensor in sb3_policy.state_dict().items():
        for prefix, new in _PREFIXES.items():
            if key.startswith(prefix):
                state[new + key[len(prefix):]] = tensor.detach().cpu().clone()
    return state


def export_policy(model, out_path, skip=4):
    """model: PPO de SB3 ya cargado/entrenado. Guarda pesos float32 + metadatos."""
    import stable_baselines3

    policy = model.policy
    c, h, w = policy.observation_space.shape  # SB3 la guarda channel-first
    assert h == w, "se asume observacion cuadrada"
    n_actions = int(policy.action_space.n)
    assert n_actions == len(SIMPLE_MOVEMENT_BUTTONS), "la tabla de acciones no coincide con el modelo"

    state = extract_actor_state(policy)
    NatureCNNPolicy(n_actions, c, h).load_state_dict(state, strict=True)  # falla si faltan/sobran claves

    meta = {
        "format_version": 1,
        "n_actions": n_actions,
        "in_channels": int(c),
        "size": int(h),
        "skip": int(skip),
        "color_order": "RGB",
        "buttons": SIMPLE_MOVEMENT_BUTTONS,
        "sb3_version": stable_baselines3.__version__,
        "torch_version": str(torch.__version__),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "meta": meta}, out_path)
    return out_path
