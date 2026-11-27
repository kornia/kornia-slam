# Evaluation scripts

Run the `kornia-slam` app over EuRoC and TUM-VI sequences and score the trajectories it writes,
so a benchmark row can be reproduced from the repository.

```bash
python -m venv .venv && .venv/bin/pip install -r scripts/eval/requirements.txt
.venv/bin/python -m unittest discover -s scripts/eval/tests
```

## Reproduce a benchmark row

```bash
# 1. TUM-VI only: convert the fisheye export to a pinhole EuRoC-layout sequence.
python scripts/eval/tumvi_to_euroc.py data/tumvi/dataset-room1_512_16 data/tumvi-pinhole/room1

# 2. Run one arm: a shipped preset plus knob lines, over one or more sequences.
cargo build --release -p kornia-slam-app
BIN=target/release/kornia-slam VOCAB=weights/ORBvoc.txt KNOBS='mapping: ( execution: Synchronous ),' \
  scripts/eval/run.sh main configs/euroc-stereo-imu-loop.ron runs data/euroc/MH_01_easy data/tumvi-pinhole/room1

# 3. Score every run directory: one JSON line each.
python scripts/eval/score.py runs/main_* > scores.jsonl

# 4. Tabulate one scored file and one metric.
python scripts/eval/table.py scores.jsonl --key kornia_slam_raw.csv@nearest --metric ate_se3_rmse
```

`run.sh` takes its settings from the environment; the header of each script lists them. `RUNS`
repeats each sequence, `JOBS` caps concurrent runs, `START_FRAME` and `MAX_FRAMES` set the frame
window, `PACE` selects the replay pacing, `CPUS` pins the run with `taskset`. A run directory whose
`done` file holds `rc=0` is skipped, so an interrupted series resumes where it stopped. `run.sh`
exits 1 when any run failed to configure, ended with a non-zero `rc` or went `invalid`.

A run directory holds `run.ron` (the run file), `meta.txt` (arm, sequence, data path, start frame,
repetition, binary md5, pacing, load), `out.log`, `err.log`, the `--evaluate` CSVs, then `done`
(`rc`, `wall`, `load_end`). The lines `err.log` prints before `Dataset:` are the system
configuration; when they differ from the arm's header (`<OUTDIR>/<ARM>.header`, copied from the
arm's first run that printed `Dataset:` and ended with `rc=0`), the run gets `invalid` in place of
`done`, because a knob did not reach it. A run that stopped before `Dataset:` has no header to
compare and keeps its `rc` in `done`.

The protocol is causal paced replay: each frame is processed once, in order, with only the data
before it, at the pace `PACE` selects (`app`: the app's own pacing, the default).

## Output keys

`score.py` writes `status`, `rc`, `wall`, `meta`, `s` (counters from `err.log`: the per-frame time
sum, accepted and applied loop closures, failed pose-graph solves, and the fields of any
`[summary]` line, which the app on main does not print), and one object per scored file. The
status is `ok`, `failed` (`rc` is not 0: a timeout or crash; its partial trajectory is scored but
kept out of tables), `invalid`, `not done`, or `error` (the run directory could not be read; the
reason is in `error`, and `score.py` exits 1 after scoring the rest). A file that cannot be scored
gets `{"error": reason}`.

| key | meaning |
|---|---|
| `pose` | `body` (TUM trajectories) or `camera_centre` (`kornia_slam_raw.csv`) |
| `n_est`, `n_assoc` | estimated poses, and those associated with a ground-truth pose |
| `total_frames`, `tracked_pct` | cam0 frames with a cam1 partner (all cam0 frames without cam1), and `n_assoc / total_frames` |
| `ate_se3_rmse`, `ate_se3_mean` | ATE (m) after SE(3) alignment, scale fixed: the primary number |
| `ate_sim3_rmse`, `ate_sim3_mean`, `sim3_scale` | ATE (m) after Sim(3) alignment, and its scale |
| `jumps` | consecutive associated pairs whose relative motion differs from ground truth by more than 0.3 m |
| `assoc_error` | present when no pose associates: evo's reason (often a clock or unit mismatch) |
| `rpe1`, `rpe10` | TUM files only: translation RMSE (m) and yaw RMSE (deg) over pose pairs 1 s and 10 s apart |

Scored files:

- `traj.txt`, `traj_smoothed.txt`, `traj_vio.txt`: body poses, written when the binary has the
  matching output flags (`run.sh` adds `--trajectory-out` when `--help` lists it).
- `kornia_slam_raw.csv`: the `--evaluate` camera centres, associated within 5 ms. Its rows carry
  no timestamps; row *i* takes the cam0 stamp of the frame named by the *i*-th status line in
  `err.log`.
- `kornia_slam_raw.csv@nearest`: the same with the nearest ground-truth pose at any distance. Its
  `ate_sim3_rmse` and `n_assoc` equal the app's own `ATE RMSE` and `Frames evaluated`.
- `<file>@crop`, with `score.py --crop-to TEMPLATE`: only the frames the trajectory at `TEMPLATE`
  covers (`{seq}` expands to the sequence), to compare with a system that drops frames.

Camera-centre and body trajectories differ by the constant cam0 lever arm, so their ATEs differ
too (by a few millimetres to centimetres on EuRoC). Never put the two in one table: `table.py`
takes one `--key` per table for this reason. Its footer counts each status, plus the ok runs with
no entry for the key or no value for the metric, which are in no median.

`evaluate.py` scores a single trajectory without a run directory:

```bash
python scripts/eval/evaluate.py data/euroc/MH_01_easy traj.txt --rpe 1,10
python scripts/eval/evaluate.py data/euroc/MH_01_easy kornia_slam_raw.csv --fmt kornia --log err.log --nearest-gt
```

## TUM-VI

The 512×512 16-bit exports in the EuRoC layout are at

```text
https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-<name>_512_16.tar
```

for example `dataset-room1_512_16`. `tumvi_to_euroc.py` undistorts both cameras from the
equidistant fisheye model to a zero-distortion pinhole (`--focal 190`, `--size 512`, principal
point at the image centre, same orientation) with a bilinear remap of the 16-bit image followed by
its high byte, writes `sensor.yaml` with `T_BS = T_cam_imu⁻¹`, copies `imu0`, and copies `mocap0` to
`state_groundtruth_estimate0`. The stereo-inertial EuRoC presets then run on it unchanged.

At f = 190 px a 512 px image spans ±53° (atan(255.5 / 190)) across each axis, much less than the
fisheye's field of view: the converted sequence drops the image periphery, so its numbers are not
comparable with runs on the raw fisheye images (`configs/tumvi-imu.ron`).
