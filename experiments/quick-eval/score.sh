#!/bin/bash
# Score run dirs: pose counts, exit code, IMU mode, [summary] counts, SE3 ATE / Sim3 scale, and the inertial-init lines.
#   score.sh OUTDIR [GLOB]   (run dir names must start with r01/r03/r11, or set SEQ)
# Each run line prints: poses=<rows in traj.txt> poses_written=<[summary] poses_written> exit=<app exit code>
# imu=<raw|rectified, from the Dataset line> then the counters and ATE. A value that is missing prints as "?",
# a missing ground truth as ate=NA(...), and an ATE failure prints its last stderr line.
# Env: LAMARIA_DATA (as in make_config.sh).
here=$(cd "$(dirname "$0")" && pwd)
for d in "$1"/${2:-*}; do n=$(basename "$d"); e=$d/stderr.txt; [ -f "$e" ] || continue
  case ${SEQ:-$n} in r01*|R_01*) q=R_01_easy;; r03*|R_03*) q=R_03_easy;; r11*|R_11*) q=R_11_5cp;; *) q=${SEQ:-};; esac
  sum=$(grep -h "^\[summary\] frames" "$e" | grep -oE "tracked=[0-9]+|coast_inertial=[0-9]+|give_ups=[0-9]+" | tr '\n' ' ')
  poses=$( [ -f "$d/traj.txt" ] && grep -cvE '^[[:space:]]*(#|$)' "$d/traj.txt" || echo '?')
  written=$(grep -ho "^\[summary\] poses_written=[0-9]*" "$e" | tail -1 | cut -d= -f2); written=${written:-?}
  code=$( [ -f "$d/stdout.txt" ] && grep -o "^exit [0-9]*" "$d/stdout.txt" | tail -1 | cut -d' ' -f2 ); code=${code:-?}
  imu=$(grep -ho "IMU [a-z]*, offset" "$e" | head -1 | cut -d' ' -f2 | tr -d ,); imu=${imu:-?}
  gt=$LAMARIA_DATA/$q/ground_truth/pseudo_dense/$q.txt
  if [ -z "$q" ] || [ ! -f "$gt" ]; then ate="ate=NA(no pseudo-dense ground truth: ${q:-no sequence})"
  elif [ ! -f "$d/traj.txt" ]; then ate="ate=NA(no traj.txt)"
  else
    out=$(python3 -I "$here/ate.py" "$d/traj.txt" "$gt" "${MAX_FRAMES:-800}" 2>&1); rc=$?
    ate=$(printf '%s\n' "$out" | grep -oE "ate_se3=[0-9.]+m|scale=[0-9.]+" | tr '\n' ' ')
    [ "$rc" -eq 0 ] && [ -n "$ate" ] || ate="ate=NA(ate.py exit $rc: $(printf '%s\n' "$out" | tail -1 | cut -c1-120))"
  fi
  echo "== $n  poses=$poses poses_written=$written exit=$code imu=$imu  $sum $ate"
  grep -hE "gate dropped|VIBA[012] (accepted|rejected)|implausible" "$e" | sed 's/ gyro_bias.*//' | cut -c1-160 | sed 's/^/    /'
done
