#!/bin/bash
# Run one arm (a preset plus knob lines) over sequences, RUNS times each, at most JOBS at once.
#   run.sh ARM PRESET.ron OUTDIR SEQ_DIR [SEQ_DIR ...]
# Each run writes OUTDIR/<ARM>_<seq>_s<start>_r<k>/ with run.ron, meta.txt, out.log, err.log,
# header.txt, the --evaluate CSVs and traj.txt (when the binary has --trajectory-out), then
# `done` (rc, wall, load) or `invalid` (header guard). A directory whose `done` holds rc=0 is skipped.
# Exits 1 when any run failed to configure, ended with rc != 0 or went invalid; 2 on a usage error.
# Env: BIN (kornia-slam binary, required), RUNS (1), JOBS (1), START_FRAME (0), MAX_FRAMES (0), VOCAB,
#      KNOBS (newline-separated knob lines), PACE=app (no flag: the app default) | none (--no-pace) |
#      timestamps (--pace-timestamps), CPUS (taskset CPU list), TIMEOUT (900 s), DEBUG (0 | 1: --debug
#      and KORNIA_SLAM_DEBUG=1), EVALUATE (1: --evaluate), RUN_FLAGS (extra app flags).
set -u
if [ $# -lt 4 ]; then
  echo "usage: run.sh ARM PRESET.ron OUTDIR SEQ_DIR [SEQ_DIR ...]" >&2; exit 2
fi
arm=$1 preset=$2 out=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
: "${BIN:?set BIN to the kornia-slam binary}"
[ -f "$BIN" ] && [ -x "$BIN" ] || { echo "run.sh: BIN=$BIN is not an executable file" >&2; exit 2; }
start=${START_FRAME:-0}
help=$("$BIN" --help 2>&1) || { echo "run.sh: $BIN --help failed: $help" >&2; exit 2; }
has_flag() { grep -q -- "$1" <<< "$help"; }
case ${PACE:-app} in
  app) pace_flag="" ;;
  none) pace_flag=--no-pace ;;
  timestamps) pace_flag=--pace-timestamps ;;
  *) echo "run.sh: PACE must be app, none or timestamps, got '${PACE}'" >&2; exit 2 ;;
esac
if [ -n "$pace_flag" ] && ! has_flag "$pace_flag"; then
  echo "run.sh: PACE=${PACE} needs $pace_flag, which $BIN does not list" >&2; exit 2
fi
case ${DEBUG:-0} in
  1) debug_flag=--debug; export KORNIA_SLAM_DEBUG=1 ;;
  # The library treats KORNIA_SLAM_DEBUG set to anything as on.
  0) debug_flag=""; unset KORNIA_SLAM_DEBUG ;;
  *) echo "run.sh: DEBUG must be 0 or 1, got '${DEBUG}'" >&2; exit 2 ;;
esac
traj_out=0; has_flag --trajectory-out && traj_out=1
taskset_cmd=(); [ -n "${CPUS:-}" ] && taskset_cmd=(taskset -c "$CPUS")
knobs=(); [ -n "${KNOBS:-}" ] && mapfile -t knobs <<< "$KNOBS"
md5=$(md5sum "$BIN" | cut -d' ' -f1)
mkdir -p "$out"
# Background jobs cannot set a variable here, so each failed run appends its directory.
failed=$(mktemp); trap 'rm -f "$failed"' EXIT
load() { cut -d' ' -f1-3 /proc/loadavg; }

one() {
  local seq_dir=$1 k=$2 seq d flags s0 rc wall
  seq=$(basename "$seq_dir")
  d=$out/${arm}_${seq}_s${start}_r$k
  grep -qs '^rc=0 ' "$d/done" && return 0
  rm -rf "$d"; mkdir -p "$d"
  # meta.txt first: a run that fails to configure still scores as `not done`.
  printf '%s\n' "arm=$arm" "seq=$seq" "data=$(cd "$seq_dir" && pwd)" "start=$start" "rep=$k" \
    "bin_md5=$md5" "pace=${PACE:-app}" "load_start=$(load)" > "$d/meta.txt"
  if ! "$here/make_config.sh" "$preset" "$seq_dir" "$d/run.ron" "${knobs[@]}"; then
    echo "$d (make_config.sh)" >> "$failed"; return
  fi
  flags=(--config "$d/run.ron" --no-tui)
  [ -n "$pace_flag" ] && flags+=("$pace_flag")
  [ -n "$debug_flag" ] && flags+=("$debug_flag")
  [ "${EVALUATE:-1}" = 1 ] && flags+=(--evaluate --eval-out "$d")
  [ "$traj_out" = 1 ] && flags+=(--trajectory-out "$d/traj.txt")
  s0=$(date +%s)
  # shellcheck disable=SC2086
  timeout "${TIMEOUT:-900}" "${taskset_cmd[@]}" "$BIN" "${flags[@]}" ${RUN_FLAGS:-} > "$d/out.log" 2> "$d/err.log"
  rc=$?; wall=$(( $(date +%s) - s0 ))
  # The lines before `Dataset:` print the system configuration: equal across one arm's runs. A run
  # that died before `Dataset:` has no header, only its error, so it neither seeds nor meets the guard.
  if grep -q '^Dataset:' "$d/err.log"; then
    sed '/^Dataset:/,$d' "$d/err.log" > "$d/header.txt"
    [ -s "$out/$arm.header" ] || [ "$rc" -ne 0 ] || cp "$d/header.txt" "$out/$arm.header"
    if [ -s "$out/$arm.header" ] && ! hd=$(diff -u "$out/$arm.header" "$d/header.txt"); then
      { echo "$hd"; echo "rc=$rc wall=$wall"; } > "$d/invalid"
      echo "$d (invalid header)" >> "$failed"
      echo "run.sh: INVALID header $d" >&2; return
    fi
  fi
  echo "rc=$rc wall=$wall load_end=$(load)" > "$d/done"
  [ "$rc" -eq 0 ] || echo "$d (rc=$rc)" >> "$failed"
  echo "run.sh: $d rc=$rc wall=${wall}s" >&2
}

for seq_dir in "$@"; do
  for k in $(seq 1 "${RUNS:-1}"); do
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-1}" ]; do wait -n; done
    one "$seq_dir" "$k" &
  done
done
wait
if [ -s "$failed" ]; then
  echo "run.sh: $(wc -l < "$failed") runs failed:" >&2; cat "$failed" >&2; exit 1
fi
