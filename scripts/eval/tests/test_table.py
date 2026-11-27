import json
import os
import subprocess
import sys
import tempfile
import unittest

TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'table.py')
KEY = 'kornia_slam_raw.csv@nearest'


def rec(seq, arm, status='ok', ate=None, key=KEY):
    r = {'run': f'{arm}_{seq}', 'arm': arm, 'seq': seq, 'status': status}
    if ate is not None:
        r[key] = {'ate_se3_rmse': ate}
    return r


class TableTest(unittest.TestCase):
    def table(self, records, *args):
        with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
            f.write(''.join(json.dumps(r) + '\n' for r in records))
        try:
            return subprocess.run([sys.executable, TABLE, f.name, '--key', KEY, *args],
                                  capture_output=True, text=True)
        finally:
            os.unlink(f.name)

    def test_median_over_ok_runs_only(self):
        p = self.table([rec('MH_01', 'main', ate=0.1), rec('MH_01', 'main', ate=0.3),
                        rec('MH_01', 'main', ate=0.2), rec('MH_01', 'main', 'failed', ate=9.0)])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('| MH_01 | 0.2 (0.1-0.3, 3) |', p.stdout)
        self.assertIn('ok 3, ok without key 0, ok without metric 0, failed 1', p.stdout)

    def test_footer_counts_ok_runs_without_key_or_metric(self):
        p = self.table([rec('MH_01', 'main', ate=0.1), rec('MH_01', 'main'),
                        {**rec('MH_01', 'main'), KEY: {'n_assoc': 2}},
                        {'run': 'x', 'status': 'error', 'error': 'no meta.txt'},
                        rec('V1_02', 'main', 'invalid'), rec('V1_02', 'main', 'not done')])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('| MH_01 | 0.1 (0.1-0.1, 1) |', p.stdout)
        self.assertIn('| V1_02 | - |', p.stdout)
        self.assertIn('ok 3, ok without key 1, ok without metric 1, failed 0, invalid 1, not done 1, error 1',
                      p.stdout)

    def test_unknown_key_exits_1(self):
        p = self.table([rec('MH_01', 'main', ate=0.1, key='traj.txt')])
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"no record has key '{KEY}'", p.stderr)


if __name__ == '__main__':
    unittest.main()
