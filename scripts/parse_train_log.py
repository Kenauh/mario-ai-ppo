"""Tablas de SB3 en el stdout de train.py -> CSV. Uso: python scripts/parse_train_log.py <log> <salida.csv>"""
import csv
import re
import sys

KEYS = ["total_timesteps", "fps", "ep_len_mean", "ep_rew_mean", "mean_max_x", "best_max_x", "flag_rate",
        "entropy_loss", "value_loss", "approx_kl", "clip_fraction", "explained_variance",
        "policy_gradient_loss", "loss"]
PAT = re.compile(r"^\|\s+([A-Za-z_]+)\s+\|\s+(\S+)\s+\|")


def main():
    rows, cur = [], {}
    for line in open(sys.argv[1], errors="replace"):
        m = PAT.match(line)
        if not m:
            continue
        try:
            cur[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
        if m.group(1) == "value_loss":
            rows.append(cur)
            cur = {}
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KEYS)
        for r in rows:
            w.writerow([r.get(k, "") for k in KEYS])
    print(len(rows), "tablas ->", sys.argv[2])


if __name__ == "__main__":
    main()
