#!/bin/bash
# Derive a run file from a shipped preset, with optional knob lines.
#   make_config.sh PRESET.ron SEQ_DIR OUT.ron ['knob line inside OrbSlam((...))' ...]
# The preset's one data: path becomes SEQ_DIR (absolute), followed by start_frame and max_frames;
# a vocabulary: path becomes $VOCAB. Knob lines go right after `pipeline: OrbSlam((`.
# Env: VOCAB (ORBvoc path, required when the preset has a vocabulary), START_FRAME (0), MAX_FRAMES (0).
# Exit 2 on a usage error, 3 unless exactly one data: path was replaced.
set -euo pipefail
if [ $# -lt 3 ]; then
  echo "usage: make_config.sh PRESET.ron SEQ_DIR OUT.ron ['knob line' ...]" >&2; exit 2
fi
preset=$1 out=$3
seq=$(cd "$2" 2>/dev/null && pwd) || { echo "make_config.sh: no sequence directory $2" >&2; exit 2; }
shift 3
vocab=""
[ -n "${VOCAB:-}" ] && vocab=$(realpath -m "$VOCAB")
rc=0
# Values go through ENVIRON: awk -v would expand backslash escapes in paths and knob lines.
MC_DATA=$seq MC_VOCAB=$vocab MC_START=${START_FRAME:-0} MC_MAXF=${MAX_FRAMES:-0} \
  MC_KNOBS=$(printf '%s\n' "$@") awk '
  BEGIN {
    data = ENVIRON["MC_DATA"]; vocab = ENVIRON["MC_VOCAB"]; start = ENVIRON["MC_START"]
    maxf = ENVIRON["MC_MAXF"]; knobs = ENVIRON["MC_KNOBS"]
  }
  # Replaces the first match of re in s with literal text r (sub() would expand & in paths).
  function put(s, re, r) { return match(s, re) ? substr(s, 1, RSTART - 1) r substr(s, RSTART + RLENGTH) : s }
  {
    # Only the code before a // comment is rewritten: the presets name fields in their comments too.
    c = index($0, "//"); code = c ? substr($0, 1, c - 1) : $0; note = c ? substr($0, c) : ""
    had = code
    while (match(code, /(^|[^A-Za-z0-9_])(start_frame|max_frames):[[:space:]]*[0-9]+[[:space:]]*,?[[:space:]]*/)) {
      lead = substr(code, RSTART, 1); if (lead ~ /[A-Za-z0-9_]/) lead = ""
      code = substr(code, 1, RSTART - 1) lead substr(code, RSTART + RLENGTH)
    }
    if (code != had && code ~ /^[[:space:]]*$/ && note == "") next
    done_part = ""
    while (match(code, /(^|[^A-Za-z0-9_])data:[[:space:]]*"[^"]*"/)) {
      head = substr(code, 1, RSTART + RLENGTH - 1); code = substr(code, RSTART + RLENGTH)
      done_part = done_part put(head, "data:[[:space:]]*\"[^\"]*\"$", "data: \"" data "\", start_frame: " start ", max_frames: " maxf)
      n_data++
    }
    code = done_part code
    if (code ~ /vocabulary:[[:space:]]*"/) {
      if (vocab == "") { missing_vocab = 1 }
      code = put(code, "vocabulary:[[:space:]]*\"[^\"]*\"", "vocabulary: \"" vocab "\"")
    }
    print code note
    if (code ~ /pipeline: OrbSlam\(\(/ && knobs != "") {
      n = split(knobs, k, "\n"); for (i = 1; i <= n; i++) if (k[i] != "") print "            " k[i]
      placed = 1
    }
  }
  END {
    if (missing_vocab) { print "make_config.sh: the preset names a vocabulary; set VOCAB" > "/dev/stderr"; exit 2 }
    if (knobs != "" && !placed) { print "make_config.sh: knob lines given but the preset has no `pipeline: OrbSlam((` line" > "/dev/stderr"; exit 2 }
    if (n_data != 1) { print "make_config.sh: replaced " n_data + 0 " data: paths, expected 1" > "/dev/stderr"; exit 3 }
  }
' "$preset" > "$out" || rc=$?
if [ "$rc" -ne 0 ]; then rm -f "$out"; exit "$rc"; fi
