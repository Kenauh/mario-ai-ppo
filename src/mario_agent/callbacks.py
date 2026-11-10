"""Callbacks de entrenamiento: registro de episodios y progreso en CSV (y TensorBoard)."""
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
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
            if "episode" in info:   # recompensa SIN escalar (la registra Monitor)
                self._ep_r[i] = float(info["episode"]["r"])
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


EVAL_FIELDS = ["timesteps", "det_max_x", "det_frames", "det_cause", "det_flag", "new_best"]


class DetEvalCallback(BaseCallback):
    """Evalua en modo DETERMINISTA (argmax), como lo haria el evaluador, cada 'eval_every' pasos.

    El emulador es determinista, asi que 1 episodio desde el reset basta. Guarda eval.csv y el
    mejor modelo (bandera > max_x > menos fotogramas) en best.zip y best_policy.pt.
    """

    def __init__(self, run_dir, models_dir, env_kwargs, eval_every=100_000):
        super().__init__()
        self.run_dir, self.models_dir = Path(run_dir), Path(models_dir)
        self.env_kwargs, self.eval_every = dict(env_kwargs), int(eval_every)

    def _on_training_start(self):
        from .env_factory import make_mario_env
        self._env = make_mario_env(**self.env_kwargs)
        self._last = self.num_timesteps
        self._f, self._w = _open_csv(self.run_dir / "eval.csv", EVAL_FIELDS)
        self._best_file = self.models_dir / "best_score.json"
        self._best = tuple(json.loads(self._best_file.read_text())) if self._best_file.exists() else None
        if self.num_timesteps == 0:
            self._evaluate()

    def _evaluate(self):
        env, policy = self._env, self.model.policy
        was_training = policy.training
        policy.set_training_mode(False)
        obs, _ = env.reset(seed=0)
        max_x, info, term, trunc = 0, {}, False, False
        while not (term or trunc):
            x = torch.as_tensor(obs[None]).permute(0, 3, 1, 2).contiguous().to(self.model.device)
            with torch.no_grad():
                logits = policy.get_distribution(x).distribution.logits
            obs, _, term, trunc, info = env.step(int(logits.argmax(dim=1).item()))
            max_x = max(max_x, int(info.get("x_pos", 0)))
        policy.set_training_mode(was_training)
        flag = bool(info.get("flag_get", False))
        cause = "bandera" if flag else ("limite" if trunc and not term else "muerte")
        frames = int(info.get("frames", -1))
        score = (int(flag), max_x, -frames)
        new_best = self._best is None or score > self._best
        if new_best:
            from .export import export_policy
            self._best = score
            self.model.save(self.models_dir / "best")
            export_policy(self.model, self.models_dir / "best_policy.pt")
            self._best_file.write_text(json.dumps(list(score)))
        self._w.writerow([self.num_timesteps, max_x, frames, cause, int(flag), int(new_best)])
        self._f.flush()
        self.logger.record("eval/det_max_x", max_x)
        self.logger.record("eval/det_flag", int(flag))
        print(f"[eval determinista] pasos={self.num_timesteps} max_x={max_x} fotogramas={frames} "
              f"causa={cause}{' (NUEVO MEJOR)' if new_best else ''}")

    def _on_step(self):
        if self.num_timesteps - self._last >= self.eval_every:
            self._last = self.num_timesteps
            self._evaluate()
        return True

    def _on_training_end(self):
        self._env.close()
        self._f.close()
