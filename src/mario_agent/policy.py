"""Red de la política en PyTorch puro (sin SB3). Replica la CnnPolicy de SB3 (NatureCNN)."""
import torch
import torch.nn as nn


class NatureCNNPolicy(nn.Module):
    """Entrada: uint8 (B, C, H, W). Salida: logits (B, n_actions).

    La normalización /255 va DENTRO de la red, igual que en SB3.
    """

    def __init__(self, n_actions=7, in_channels=4, size=84, features_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, in_channels, size, size)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.action_net = nn.Linear(features_dim, n_actions)

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.float() / 255.0
        return self.action_net(self.linear(self.cnn(x)))


def load_policy(path, device="cpu"):
    """Carga un .pt exportado. Devuelve (red en modo eval, metadatos)."""
    blob = torch.load(path, map_location=device, weights_only=True)
    meta = blob["meta"]
    net = NatureCNNPolicy(meta["n_actions"], meta["in_channels"], meta["size"])
    net.load_state_dict(blob["state_dict"], strict=True)
    net.to(device).eval()
    return net, meta
