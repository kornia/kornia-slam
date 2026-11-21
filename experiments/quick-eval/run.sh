#!/bin/bash
# Run configs N times each, at most JOBS at once (laptop limit 3).
#   run.sh BIN OUTDIR RUNS CONFIG.ron [CONFIG.ron ...]   -> OUTDIR/<config>-r<k>/{traj.txt,stdout.txt,stderr.txt}
# Env: RUN_FLAGS, extra app flags (e.g. --trajectory-include-coasted for benchmark scoring).
set -u
bin=$1 out=$2 runs=$3; shift 3; jobs_max=${JOBS:-3}
one() { d=$out/$1; mkdir -p "$d"; KORNIA_SLAM_DEBUG=1 "$bin" --config "$2" --no-tui --no-pace --debug \
  --trajectory-out "$d/traj.txt" ${RUN_FLAGS:-} > "$d/stdout.txt" 2> "$d/stderr.txt"; echo "exit $?" >> "$d/stdout.txt"; }
for c in "$@"; do for r in $(seq 1 "$runs"); do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do wait -n; done
  one "$(basename "$c" .ron)-r$r" "$c" &
done; done; wait
