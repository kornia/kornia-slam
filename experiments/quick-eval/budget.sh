#!/bin/bash
# budget.sh ARM SEQ ROUND: CP budget row of one gate run (card AG). Columns: CP actual, then oracle.py CPs:
#   O1 segments linked (splits at session segment starts), O3 O1 plus event times, O4-60 / O4-30 every 60 / 30 s,
#   O5 one piece with MODE=scale (global scale only), O6 O1 with coasted poses removed, and the GT-as-estimate ceiling.
# O-numbers are upper bounds: aligning each piece to GT also removes that piece's offset.
# RUN=dir-or-traj and TAG=name override the gate run (self-tests). One evaluator at a time, nice 19, one thread;
# waits while the 1-minute load is over MAXLOAD (default 8) so the gate's samples stay quiet.
set -u
Q=$(cd "$(dirname "$0")" && pwd)
AG=/home/edgar/dev/split/.pr180-fix/cardAG; G=${G:-/home/edgar/dev/split/.pr180-fix/gateH0/out}
D=/home/edgar/dev/.flux-connect-scratch/bench/lamaria/data
ARM=$1; SEQ=$2; ROUND=$3; RUN=${RUN:-$G/$ARM-$SEQ-r$ROUND}; TAG=${TAG:-$ARM-$SEQ-r$ROUND}
W=$AG/budget/work/$TAG; OUT=$AG/budget/$TAG.txt; mkdir -p "$W"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 19 python3 -I -B"
if [ -d "$RUN" ]; then TRAJ=$RUN/traj.txt; else TRAJ=$RUN; fi

quiet() { while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge "${MAXLOAD:-8}" ]; do sleep 30; done; }
cp_of() { quiet; "$Q/evalcp.sh" "$SEQ" "$1" "$2" > "$2.log" 2>&1; grep -m1 -oE 'CP Score: [0-9.]+' "$2.log" | awk '{print $3}'; }

if [ -f "$RUN/cp_online.txt" ]; then
  CP=$(grep -m1 -oE 'CP Score: [0-9.]+' "$RUN/cp_online.txt" | awk '{print $3}')
else
  CP=$(cp_of "$TRAJ" "$W/actual")
fi
CP=${CP:-fail}
if [ ! -s "$D/$SEQ/ground_truth/pseudo_dense/$SEQ.txt" ]; then
  echo "$TAG CP $CP (no pseudo-dense GT: oracles not computed)" | tee "$OUT"; exit 0
fi

CEIL_F=$AG/budget/ceiling-$SEQ.txt
[ -s "$CEIL_F" ] || cp_of "$D/$SEQ/ground_truth/pseudo_dense/$SEQ.txt" "$W/ceiling" > "$CEIL_F"
CEIL=$(cat "$CEIL_F")

$PY "$Q/driftbudget.py" "$RUN" "$SEQ" --emit "$W/splits" > "$W/driftbudget.txt"
oracle() {  # NAME SPLITS [TRAJ]
  $PY "$Q/oracle.py" "$SEQ" "$2" "$W/$1.traj" "${3:-$TRAJ}" > "$W/$1.oracle.txt"
  cp_of "$W/$1.traj" "$W/$1"
}
O1=$(oracle O1 "$(cat "$W/splits/O1.txt")")
O3=$(oracle O3 "$(cat "$W/splits/O3.txt")")
O460=$(oracle O4-60 "$(cat "$W/splits/O4-60.txt")")
O430=$(oracle O4-30 "$(cat "$W/splits/O4-30.txt")")
O5=$(MODE=scale oracle O5 "")
awk -v C="$W/splits/coasted.txt" 'BEGIN { while ((getline x < C) > 0) c[x] = 1 } !($1 in c)' "$TRAJ" > "$W/nocoast.txt"
O6=$(oracle O6 "$(cat "$W/splits/O1.txt")" "$W/nocoast.txt")
echo "$TAG CP $CP O1 ${O1:-fail} O3 ${O3:-fail} O4-60 ${O460:-fail} O4-30 ${O430:-fail} O5 ${O5:-fail} O6 ${O6:-fail} ceiling ${CEIL:-fail}" | tee "$OUT"
