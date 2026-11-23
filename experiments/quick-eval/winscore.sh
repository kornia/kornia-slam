#!/usr/bin/env bash
# winscore.sh RUN_DIR SEQ LO HI: one window of a run, scored by the existing tools (card AH).
# LO/HI are s from the sequence's first cam0 frame. Prints ate.py on the window cut, the largest 30 s piece yaw step
# from driftbudget.py, worldrot.py jumps, biasevent.py --window (SUMMARY/EVENT lines), the [summary] lines and the
# wall time (wall.txt in the run dir, if the runner wrote one).
set -euo pipefail
run=$1 seq=$2 lo=$3 hi=$4
Q=$(cd "$(dirname "$0")" && pwd)
D=/home/edgar/dev/.flux-connect-scratch/bench/lamaria/data
py() { nice -n 19 python3 -I -B "$@"; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
t0=$(awk -F, '!/^#/ {print $1; exit}' "$D/$seq/asl/$seq/aria/cam0/data.csv")
cut=$tmp/traj.txt
awk -v t0="$t0" -v lo="$lo" -v hi="$hi" '!/^#/ { t = ($1 - t0) / 1e9; if (t >= lo && t < hi) print }' "$run/traj.txt" > "$cut"
nf=$(awk -F, -v t0="$t0" -v lo="$lo" -v hi="$hi" '!/^#/ { t = ($1 - t0) / 1e9; if (t >= lo && t < hi) n++ } END { print n + 0 }' \
  "$D/$seq/asl/$seq/aria/cam0/data.csv")
echo "== $run $seq $lo:$hi ($nf cam0 frames, $(wc -l < "$cut") poses)"
echo -n "ate: "; py "$Q/ate.py" "$cut" "$D/$seq/ground_truth/pseudo_dense/$seq.txt" "$nf"
py "$Q/driftbudget.py" "$cut" "$seq" | awk '
  /^-- / { blk = $0 }
  blk ~ /trajectory/ && /^   30 s pieces/ { on = 1; next }
  blk ~ /trajectory/ && /^   (10|60|120) s pieces|^   events/ { on = 0 }
  on && NF >= 9 { y = $6; if (n++ && (d = y - prev) * d > best * best) { best = d; at = $1 } prev = y }
  /^   events/ && blk ~ /trajectory/ { ev = $0 }
  END { printf "yaw30: largest 30 s piece step %+.1f deg at %s s\n%s\n", best, at, ev }'
echo -n "jumps: "; py "$Q/worldrot.py" jumps "$cut" | head -1
py "$Q/biasevent.py" "$run" "$D/$seq" --window "$lo:$hi" | grep -E '^(SUMMARY|EVENT)'
grep -h '^\[summary\]' "$run/stderr.txt" | grep -E 'frames=|loss |frame_ms' || true
echo "wall: $(cat "$run/wall.txt" 2>/dev/null || echo -)"
