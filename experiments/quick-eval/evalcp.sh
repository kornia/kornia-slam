#!/bin/bash
# evalcp.sh SEQ EST OUTDIR: official LaMAria CP score of EST (one light ceres run); prints CP score and R@1m.
L=/home/edgar/dev/.flux-connect-scratch/bench/lamaria; SEQ=$1
cd $L/repo
OMP_NUM_THREADS=1 nice -n 19 $L/px/.pixi/envs/default/bin/python -m evaluate_wrt_control_points --estimate $2 --cp_json_file $L/data/$SEQ/ground_truth/sparse/$SEQ.json --device_calibration_json $L/data/$SEQ/aria_calibrations/$SEQ.json --corresponding_sensor imu --output_path $3 2>&1 | tr "\r" "\n" | grep -E "CP Score|Recall @ 1m|rror"
