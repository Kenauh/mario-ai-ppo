#!/usr/bin/env bash
# Reporte compacto de estado. Uso: bash scripts/state_report.sh > ~/state_report.txt 2>&1
cd "$(dirname "${BASH_SOURCE[0]}")/.." && source .venv/bin/activate
echo "## FECHA"; date
echo "## GIT"; git status --short | head -15; git log --oneline | head -4
git fetch -q 2>&1 | head -2; git status -sb | head -1
echo "## INTEGRIDAD"
for n in cfg1_base cfg2_reward_scale; do
  echo "-- $n"; ls -t models/$n | head -3
  for z in $(ls -t models/$n/ckpt_*.zip | head -1) models/$n/final.zip models/$n/best.zip; do
    printf "%s: " "$z"; python -m zipfile -t "$z" 2>&1 | tail -1
  done
  for f in episodes progress eval; do printf "%s.csv filas=" $f; wc -l < runs/$n/$f.csv; done
  awk -F, 'NF!=9{c++} END{print "episodes con campos raros:", c+0}' runs/$n/episodes.csv
  tail -1 runs/$n/progress.csv
done
ls -lh results/videos | tail -12
echo "## RESULTADOS"
for n in cfg1_base cfg2_reward_scale; do
  echo "== $n"; python scripts/monitor.py $n 2>&1 | head -4; cat models/$n/best_score.json; echo
  awk -F, 'NR>1{n++; if($5==1)f++} END{print "evals deterministas:", n, "con bandera:", f+0}' runs/$n/eval.csv
  tail -6 runs/$n/eval.csv
  echo "entropia (paso, valor) cada ~250 tablas:"
  awk -F'|' '/total_timesteps/{t=$3+0} /entropy_loss/{k++; if(k%250==1) print t, $3+0}' runs/${n%%_*}_stdout.log
done
