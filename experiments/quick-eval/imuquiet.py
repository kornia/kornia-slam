#!/usr/bin/env python3
"""IMU chunks against the noise model, as the still run judges them (card AB-1b).

  imuquiet.py SEQ_DIR --spans A-B[,C-D...] [--chunk 0.5]

SEQ_DIR is a EuRoC sequence (holds mav0/imu0/data.csv, mav0/imu0/sensor.yaml and
mav0/cam0/data.csv). --spans are in seconds from the first camera frame, as still.py's --quiet.

Each span is cut into --chunk second chunks. Per chunk it prints the gyro and accel per-axis
sample variance over the model (density^2 * rate) and whether both sit inside the band
1 + 3.09 * sqrt(2 / (3 (n - 1))), as ChunkMean::new (inertial/still.rs) does. Ends with
quiet=<chunks quiet>/<chunks> per span and in total.
"""
import argparse
import os
import re

Z_999 = 3.09


def span(text):
    a, b = text.split("-")
    return float(a), float(b)


def read_rows(csv):
    rows = []
    with open(csv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split(",")
            rows.append((int(v[0]) * 1e-9, [float(x) for x in v[1:7]]))
    return rows


def read_density(yaml, key):
    with open(yaml) as f:
        for line in f:
            if m := re.match(rf"\s*{key}:\s*([0-9.eE+-]+)", line):
                return float(m[1])
    raise SystemExit(f"{key} not in {yaml}")


def rms_dev(vecs):
    n = len(vecs)
    mean = [sum(v[i] for v in vecs) / n for i in range(3)]
    sq = sum(x * x for v in vecs for x in v) / n
    return max(sq - sum(m * m for m in mean), 0.0) ** 0.5


def judge(chunk, gyro_density, accel_density):
    n = len(chunk)
    span_sec = chunk[-1][0] - chunk[0][0]
    rate = (n - 1) / span_sec
    band = 1.0 + Z_999 * (2.0 / (3.0 * (n - 1))) ** 0.5
    ratio = lambda dev, density: (dev * dev / 3.0) / (density * density * rate)
    gyro = ratio(rms_dev([s[0:3] for _, s in chunk]), gyro_density)
    accel = ratio(rms_dev([s[3:6] for _, s in chunk]), accel_density)
    return n, band, gyro, accel, gyro <= band and accel <= band


def main():
    p = argparse.ArgumentParser()
    p.add_argument("seq_dir")
    p.add_argument("--spans", required=True)
    p.add_argument("--chunk", type=float, default=0.5)
    a = p.parse_args()

    mav = os.path.join(a.seq_dir, "mav0")
    yaml = os.path.join(mav, "imu0", "sensor.yaml")
    gyro_density = read_density(yaml, "gyroscope_noise_density")
    accel_density = read_density(yaml, "accelerometer_noise_density")
    imu = read_rows(os.path.join(mav, "imu0", "data.csv"))
    with open(os.path.join(mav, "cam0", "data.csv")) as f:
        t0 = next(int(l.split(",")[0]) * 1e-9 for l in f if l.strip() and not l.startswith("#"))

    total_quiet = total = 0
    for s, e in (span(x) for x in a.spans.split(",") if x):
        quiet = chunks = 0
        c = s
        while c + a.chunk <= e + 1e-9:
            chunk = [r for r in imu if c <= r[0] - t0 < c + a.chunk]
            if len(chunk) >= 2:
                n, band, g, acc, q = judge(chunk, gyro_density, accel_density)
                print(f"chunk t={c:.2f} n={n} band={band:.3f} gyro={g:.2f} accel={acc:.2f} quiet={int(q)}")
                quiet += q
                chunks += 1
            c += a.chunk
        print(f"span {s}-{e} quiet={quiet}/{chunks}")
        total_quiet += quiet
        total += chunks
    print(f"quiet={total_quiet}/{total}")


if __name__ == "__main__":
    main()
