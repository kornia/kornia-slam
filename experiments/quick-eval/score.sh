#!/bin/bash
# Score run dirs: [summary] counts, SE3 ATE / Sim3 scale, and the inertial-init lines.
#   score.sh OUTDIR [GLOB]   (run dir names must start with r01/r03/r11, or set SEQ)
# Env: LAMARIA_DATA (as in make_config.sh).
here=$(cd "$(dirname "$0")" && pwd)
for d in "$1"/${2:-*}; do n=$(basename "$d"); e=$d/stderr.txt; [ -f "$e" ] || continue
  case ${SEQ:-$n} in r01*|R_01*) q=R_01_easy;; r03*|R_03*) q=R_03_easy;; r11*|R_11*) q=R_11_5cp;; *) q=${SEQ:-};; esac
  sum=$(grep -h "^\[summary\] frames" "$e" | grep -oE "tracked=[0-9]+|coast_inertial=[0-9]+|give_ups=[0-9]+" | tr '\n' ' ')
  ate=$(python3 -I "$here/ate.py" "$d/traj.txt" "$LAMARIA_DATA/$q/ground_truth/pseudo_dense/$q.txt" "${MAX_FRAMES:-800}" 2>/dev/null \
        | grep -oE "ate_se3=[0-9.]+m|scale=[0-9.]+" | tr '\n' ' ')
  echo "== $n  $sum $ate"
  grep -hE "gate dropped|VIBA[012] (accepted|rejected)|implausible" "$e" | sed 's/ gyro_bias.*//' | cut -c1-160 | sed 's/^/    /'
done
