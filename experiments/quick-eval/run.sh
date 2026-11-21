#!/bin/bash
# Run configs N times each, at most JOBS at once (laptop limit 3).
#   run.sh BIN OUTDIR RUNS CONFIG.ron [CONFIG.ron ...]   -> OUTDIR/<config>-r<k>/{traj.txt,stdout.txt,stderr.txt}
# Env: RUN_FLAGS, extra app flags (e.g. --trajectory-include-coasted for benchmark scoring).
#      PACE=none (default, today: --no-pace; scores depend on machine load) | timestamps (--pace-timestamps,
#      one frame per sensor interval, so the score does not depend on load; needs an app that has the flag)
#      | app (no pace flag: the app default).
#      DEBUG=1 (default, today: KORNIA_SLAM_DEBUG=1 and --debug) | 0 (neither, and KORNIA_SLAM_DEBUG unset because
#      the library reads it as set-to-anything: the validation run, no debug cost).
set -u
bin=$1 out=$2 runs=$3; shift 3; jobs_max=${JOBS:-3}
case ${PACE:-none} in
  none) pace_flag=--no-pace;; timestamps) pace_flag=--pace-timestamps;; app) pace_flag="";;
  *) echo "run.sh: PACE must be none, timestamps or app, got '${PACE}'" >&2; exit 2;; esac
case ${DEBUG:-1} in
  1) debug_flag=--debug; export KORNIA_SLAM_DEBUG=1;; 0) debug_flag=""; unset KORNIA_SLAM_DEBUG;;
  *) echo "run.sh: DEBUG must be 0 or 1, got '${DEBUG}'" >&2; exit 2;; esac
one() { d=$out/$1; mkdir -p "$d"; "$bin" --config "$2" --no-tui $pace_flag $debug_flag \
  --trajectory-out "$d/traj.txt" ${RUN_FLAGS:-} > "$d/stdout.txt" 2> "$d/stderr.txt"; echo "exit $?" >> "$d/stdout.txt"; }
for c in "$@"; do for r in $(seq 1 "$runs"); do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do wait -n; done
  one "$(basename "$c" .ron)-r$r" "$c" &
done; done; wait
