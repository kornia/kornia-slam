#!/usr/bin/env python3
"""table.py SCORES.jsonl [--key FILE] [--metric NAME]: markdown table of score.py output.

Rows are sequences, columns arms, cells "median (min-max, n)" of the metric over the ok runs.
One --key per table, so body and camera-centre numbers never share one.
"""
import argparse
import json
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scores')
    ap.add_argument('--key', default='traj.txt', help='scored file, e.g. kornia_slam_raw.csv@nearest')
    ap.add_argument('--metric', default='ate_se3_rmse')
    a = ap.parse_args()
    values = defaultdict(list)
    status = defaultdict(int)
    arms, seqs = [], []
    with open(a.scores) as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        arm, seq = r['arm'], r['seq']
        arms += [arm] if arm not in arms else []
        seqs += [seq] if seq not in seqs else []
        status[r['status']] += 1
        v = r.get(a.key, {}).get(a.metric) if r['status'] == 'ok' else None
        if v is not None:
            values[seq, arm].append(v)

    def cell(v):
        if not v:
            return '-'
        return f'{statistics.median(v):.4g} ({min(v):.4g}-{max(v):.4g}, {len(v)})'

    print(f'{a.key} {a.metric}\n')
    print('| sequence | ' + ' | '.join(arms) + ' |')
    print('|---' * (len(arms) + 1) + '|')
    for seq in seqs:
        print(f'| {seq} | ' + ' | '.join(cell(values[seq, arm]) for arm in arms) + ' |')
    print(f"\nok {status['ok']}, invalid {status['invalid']}, not done {status['not done']}")


if __name__ == '__main__':
    main()
