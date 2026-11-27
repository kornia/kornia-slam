#!/usr/bin/env python3
"""table.py SCORES.jsonl [--key FILE] [--metric NAME]: markdown table of score.py output.

Rows are sequences, columns arms, cells "median (min-max, n)" of the metric over the ok runs.
One --key per table, so body and camera-centre numbers never share one. The footer counts every
status, and the ok runs that have no --key entry or no --metric value: those runs are not in any
median, so a cell's n can be smaller than its ok runs. Exits 1 when no record has --key.
"""
import argparse
import json
import statistics
import sys
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
    has_key = 0
    for r in records:
        status[r['status']] += 1
        has_key += a.key in r
        if r['status'] == 'error':
            continue
        arm, seq = r['arm'], r['seq']
        arms += [arm] if arm not in arms else []
        seqs += [seq] if seq not in seqs else []
        if r['status'] != 'ok':
            continue
        if a.key not in r:
            status['ok without key'] += 1
            continue
        v = r[a.key].get(a.metric)
        if v is None:
            # an evaluate error, or fewer than 3 associated poses
            status['ok without metric'] += 1
        else:
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
    print('\n' + ', '.join(f'{k} {status[k]}' for k in
                            ('ok', 'ok without key', 'ok without metric', 'failed', 'invalid', 'not done', 'error')))
    if not has_key:
        sys.exit(f'table.py: no record has key {a.key!r}')


if __name__ == '__main__':
    main()
