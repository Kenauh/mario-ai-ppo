"""Cadena absorbente por zonas de x. Uso: python scripts/markov_zones.py <episodes.csv> [--zone 100] [--since N] [--until N]"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--zone", type=int, default=100)
ap.add_argument("--xflag", type=int, default=3161)
ap.add_argument("--since", type=int, default=0)
ap.add_argument("--until", type=int, default=10**12)
a = ap.parse_args()
e = pd.read_csv(a.csv)
e = e[(e.timesteps > a.since) & (e.timesteps <= a.until)]
z = (e.max_x // a.zone).astype(int)
K = a.xflag // a.zone + 1
rows = []
for k in range(K):
    reach = int((z >= k).sum())
    if reach == 0:
        break
    end = int((z == k).sum())
    m = int(((z == k) & (e.cause == "muerte")).sum())
    s = int(((z == k) & (e.cause == "limite_fotogramas")).sum())
    b = int(((z == k) & (e.flag_get == 1)).sum())
    rows.append(dict(zona=k * a.zone, llegan=reach, muerte=m, atasco=s, bandera=b,
                     q_muerte=round(m / reach, 3), p_sigue=round((reach - end) / reach, 3)))
t = pd.DataFrame(rows)
print(f"episodios={len(e)} banderas={int(e.flag_get.sum())} ({e.flag_get.mean() * 100:.1f} %)")
print("Zonas con mas muertes (q_muerte = P(morir en la zona dado que llego)):")
print(t.sort_values("muerte", ascending=False).head(10).to_string(index=False))
prod = t.p_sigue.iloc[:-1].prod() * (t.bandera.iloc[-1] / t.llegan.iloc[-1])
print(f"producto de p_sigue y P(bandera en la ultima zona) = {prod:.3f} (debe coincidir con la tasa de bandera)")
