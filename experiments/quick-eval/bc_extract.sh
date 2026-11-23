#!/bin/bash
# Cut one break case's T1 fixture out of a recorded run (experiments/break-cases.md).
#   bc_extract.sh CASE   -> $OUT/CASE/{session.cbor,gt.txt,frames/*.png} as the case needs, then their sizes
# Env: BIN      kornia-slam app binary with the session-slice subcommand (required for slice cases)
#      LAMARIA  LaMAria data directory holding <seq>/asl, <seq>/ground_truth (required)
#      SESSION  session.cbor of a full run of the case's sequence (required for slice cases)
#      OUT      default crates/kornia-slam/tests/fixtures/break_cases
# Spans are seconds since the sequence's first frame, as start_sec counts them; the script moves them to the
# session's clock with the first cam0 stamp. GT keeps 1 s either side of the span. Mini-trace cases (BC-01, 07, 14,
# 17) need a --trace-out capture first. A state from before an event needs no new code: a window run with end_sec
# just before the event and --session-out.
set -euo pipefail
case_id=${1:?usage: bc_extract.sh CASE}
: "${LAMARIA:?set LAMARIA to the LaMAria data directory}"
out=${OUT:-$(dirname "$0")/../../crates/kornia-slam/tests/fixtures/break_cases}/$case_id

# case -> sequence, slice span (s), frame span (s; - for none). Slices hold 10-35 keyframes and at most 6 s of IMU.
case $case_id in
  BC-02) seq=sequence_1_19 slice=- frames="437.25 437.35";;   # 3 frames before the lost keyframe at 437.4 s
  BC-03) seq=sequence_1_19 slice="586.0 589.0" frames=-;;     # worst keyframe 588.65 s, 14.71 m/s against GT 0.32
  BC-04) seq=sequence_3_18 slice="1213.6 1219.5" frames=-;;   # the 1215.60-1219.45 s coast and 2 s before it
  BC-09) seq=sequence_1_19 slice="60.0 63.0" frames=-;;       # start of the 60-70 s fixture span
  BC-15) seq=R_11_5cp slice="329.0 335.0" frames=-;;          # the fast turn, 34 keyframes
  BC-01|BC-07|BC-14|BC-17) echo "$case_id is a mini-trace: it needs a --trace-out capture (stop S2)" >&2; exit 2;;
  *) echo "bc_extract.sh: $case_id has no T1 extract" >&2; exit 2;;
esac

cam=$LAMARIA/$seq/asl/$seq/aria/cam0
gt=$LAMARIA/$seq/ground_truth/pseudo_dense/$seq.txt
t0_ns=$(head -1 "$cam/data.csv" | cut -d, -f1)
abs() { awk -v t0="$t0_ns" -v s="$1" 'BEGIN { printf "%.9f", t0 * 1e-9 + s }'; }
mkdir -p "$out"

if [ "$slice" != - ]; then
  : "${BIN:?set BIN to the kornia-slam binary}" "${SESSION:?set SESSION to the run's session.cbor}"
  set -- $slice
  # session-slice reads no run file; --config only satisfies the parser.
  "$BIN" --config unused.ron session-slice --session "$SESSION" \
    --from-sec "$(abs "$1")" --to-sec "$(abs "$2")" --out "$out/session.cbor"
  awk -v lo="$(abs "$1")" -v hi="$(abs "$2")" '$1 * 1e-9 >= lo - 1 && $1 * 1e-9 <= hi + 1' "$gt" > "$out/gt.txt"
fi

if [ "$frames" != - ]; then
  set -- $frames
  mkdir -p "$out/frames"
  awk -F, -v lo="$(abs "$1")" -v hi="$(abs "$2")" '$1 * 1e-9 >= lo - 1e-6 && $1 * 1e-9 <= hi + 1e-6 { print $2 }' \
    "$cam/data.csv" | while read -r png; do cp "$cam/data/$png" "$out/frames/"; done
fi

find "$out" -type f -printf "%s\t%p\n" | sort -k2
du -sb "$out"
