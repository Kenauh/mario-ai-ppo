"""Callbacks de entrenamiento: registro de episodios y progreso en CSV (y TensorBoard)."""
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

EP_FIELDS = ["timesteps", "env_idx", "ep_reward", "ep_decisions", "frames",
             "max_x", "flag_get", "cause", "game_time_left"]
PROG_FIELDS = ["timesteps", "unix_time", "elapsed_s", "fps", "episodes",
               "mean_reward", "mean_max_x", "best_max_x", "flag_rate"]


def _open_csv(path, fields):
    new = not path.exists()
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(fields)
        f.flush()
    return f, w


class EpisodeLogger(BaseCallback):
    """Un renglon por episodio (episodes.csv) y uno por rollout (progress.csv)."""

    def __init__(self, run_dir, window=100):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.window = window

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._ep_r = np.zeros(n)
        self._ep_len = np.zeros(n, dtype=int)
        self._max_x = np.zeros(n, dtype=int)
        self._recent = deque(maxlen=self.window)
        self._episodes = 0
        self._t0 = time.time()
        self._ts0 = self.num_timesteps
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._f_ep, self._w_ep = _open_csv(self.run_dir / "episodes.csv", EP_FIELDS)
        self._f_pr, self._w_pr = _open_csv(self.run_dir / "progress.csv", PROG_FIELDS)

    def _on_step(self):
        rewards, dones, infos = self.locals["rewards"], self.locals["dones"], self.locals["infos"]
        for i in range(len(dones)):
            info = infos[i]
            self._ep_r[i] += rewards[i]
            self._ep_len[i] += 1
            self._max_x[i] = max(self._max_x[i], int(info.get("x_pos", 0)))
            if not dones[i]:
                continue
            flag = bool(info.get("flag_get", False))
            if flag:
                cause = "bandera"
            elif info.get("TimeLimit.truncated", False):
                cause = "limite_fotogramas"
            elif info.get("time", 1) <= 0:
                cause = "reloj"
            else:
                cause = "muerte"
            self._w_ep.writerow([self.num_timesteps, i, round(float(self._ep_r[i]), 2),
                                 int(self._ep_len[i]), info.get("frames", -1), int(self._max_x[i]),
                                 int(flag), cause, info.get("time", -1)])
            self._recent.append((self._ep_r[i], self._max_x[i], float(flag)))
            self._episodes += 1
            self._ep_r[i], self._ep_len[i], self._max_x[i] = 0.0, 0, 0
        return True

    def _on_rollout_end(self):
        self._f_ep.flush()
        elapsed = time.time() - self._t0
        fps = (self.num_timesteps - self._ts0) / max(elapsed, 1e-9)
        if self._recent:
            arr = np.array(self._recent, dtype=float)
            mean_r, mean_x, best_x, flag_rate = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 1].max(), arr[:, 2].mean()
            self.logger.record("mario/mean_max_x", float(mean_x))
            self.logger.record("mario/best_max_x", float(best_x))
            self.logger.record("mario/flag_rate", float(flag_rate))
        else:
            mean_r = mean_x = best_x = flag_rate = ""
        self._w_pr.writerow([self.num_timesteps, round(time.time(), 1), round(elapsed, 1), round(fps, 1),
                             self._episodes, mean_r, mean_x, best_x, flag_rate])
        self._f_pr.flush()

    def _on_training_end(self):
        self._f_ep.close()
        self._f_pr.close()
