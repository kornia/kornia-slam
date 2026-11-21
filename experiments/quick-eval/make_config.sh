#!/bin/bash
# Derive an 800-frame LaMAria run config from a shipped preset, with optional knob lines.
#   make_config.sh PRESET SEQ OUT.ron ['knob line inside OrbSlam((...))' ...]
# Env: LAMARIA_DATA (root holding <seq>/asl/<seq> and <seq>/pinhole_calibrations),
#      LAMARIA_FACTORY (dir with <seq>.factory.json, optional), VOCAB (ORBvoc.txt path),
#      MAX_FRAMES (default 800), RAW_FISHEYE=1 (read the raw Aria fisheye images from
#      <seq>/asl_raw/<seq> with <seq>/aria_calibrations/<seq>.json; needs a Rig, ORB-only preset).
set -euo pipefail
preset=$1 seq=$2 out=$3; shift 3
: "${LAMARIA_DATA:?set LAMARIA_DATA}" "${VOCAB:?set VOCAB to ORBvoc.txt}"
factory=""
if [ -n "${LAMARIA_FACTORY:-}" ] && [ -f "$LAMARIA_FACTORY/$seq.factory.json" ]; then
  factory="        factory_calib: \"$LAMARIA_FACTORY/$seq.factory.json\","
fi
raw=""
if [ "${RAW_FISHEYE:-0}" = 1 ]; then
  raw="        images: RawFisheye(calib: \"$LAMARIA_DATA/$seq/aria_calibrations/$seq.json\", root: \"$LAMARIA_DATA/$seq/asl_raw/$seq\"),"
fi
awk -v data="$LAMARIA_DATA/$seq/asl/$seq" -v calib="$LAMARIA_DATA/$seq/pinhole_calibrations/$seq.json" \
    -v factory="$factory" -v raw="$raw" -v maxf="${MAX_FRAMES:-800}" -v vocab="$VOCAB" -v knobs="$(printf '%s\n' "$@")" '
  /^[[:space:]]*data:/          { print "        data: \"" data "\","; next }
  /^[[:space:]]*calib:/         { print "        calib: \"" calib "\","; if (factory != "") print factory; print "        max_frames: " maxf ","; if (raw != "") print raw; next }
  /^[[:space:]]*factory_calib:/ { next }
  /ORBvoc\.txt/                 { sub(/"[^"]*ORBvoc\.txt"/, "\"" vocab "\"") }
  { print }
  /pipeline: OrbSlam\(\(/ && knobs != "" { n = split(knobs, k, "\n"); for (i = 1; i <= n; i++) if (k[i] != "") print "            " k[i] }
' "$preset" > "$out"
