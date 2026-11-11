"""Resumen por ventanas: python scripts/summarize_run.py <name> [bin]"""
import sys
import pandas as pd

name = sys.argv[1]
b = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
d = f"runs/{name}"
ep = pd.read_csv(f"{d}/episodes.csv")
ev = pd.read_csv(f"{d}/eval.csv")
ep["v"] = ((ep["timesteps"] - 1) // b + 1) * b
ev["v"] = ((ev["timesteps"] - 1) // b + 1) * b
g = ep.groupby("v")
t = pd.DataFrame({
    "n": g.size(),
    "max_x_media": g["max_x"].mean().round(0),
    "mediana": g["max_x"].median(),
    "p90": g["max_x"].quantile(0.9),
    "banderas": g["flag_get"].sum(),
    "pct_bandera": (g["flag_get"].mean() * 100).round(1),
})
t["frames_bandera"] = ep[ep.flag_get == 1].groupby("v")["frames"].median()
t["det_max_x"] = ev.groupby("v")["det_max_x"].last()
print(t.to_string())
print("Causas totales:", ep["cause"].value_counts().to_dict())
print("Evaluaciones deterministas:")
print(ev.drop(columns="v").to_string(index=False))
