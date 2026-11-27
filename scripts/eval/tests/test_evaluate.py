import json
import os
import tempfile
import unittest

import numpy as np

import synth
from evaluate import evaluate
import score

N = 120
R0 = synth.rot([0.2, 0.3, 1.0], 0.7)
T0 = np.array([1.0, -2.0, 0.5])


def write(path, text):
    with open(path, 'w') as f:
        f.write(text)


class SequenceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.seq = os.path.join(self.dir, 'seq')
        self.ts = synth.make_sequence(self.seq, N)
        self.gt = synth.poses(N)

    def tearDown(self):
        self.tmp.cleanup()

    def tum(self, ps, ts=None, name='traj.txt'):
        path = os.path.join(self.dir, name)
        synth.write_tum(path, self.ts if ts is None else ts, ps)
        return path


class EvaluateTest(SequenceCase):
    def test_rigid_copy_scores_zero(self):
        out = evaluate(self.seq, self.tum(synth.moved(self.gt, R0, T0)))
        self.assertEqual(out['ate_se3_rmse'], 0.0)
        self.assertEqual(out['n_assoc'], N)
        self.assertEqual(out['sim3_scale'], 1.0)
        self.assertEqual(out['pose'], 'body')
        self.assertEqual(out['total_frames'], N)

    def test_scaled_copy_is_zero_only_under_sim3(self):
        out = evaluate(self.seq, self.tum(synth.moved(self.gt, R0, T0, scale=1.1)))
        self.assertEqual(out['ate_sim3_rmse'], 0.0)
        self.assertEqual(out['sim3_scale'], 0.90909)
        self.assertGreater(out['ate_se3_rmse'], 0.0)

    def test_association_limit(self):
        shifted = [t + 6_000_000 for t in self.ts]
        path = self.tum(self.gt, ts=shifted)
        self.assertEqual(evaluate(self.seq, path)['n_assoc'], 0)
        self.assertEqual(evaluate(self.seq, path, max_diff=0.01)['n_assoc'], N)

    def test_mocap_ground_truth_fallback(self):
        mocap = os.path.join(self.dir, 'mocap')
        synth.make_sequence(mocap, N, gt_dir='mocap0')
        path = self.tum(synth.moved(self.gt, R0, T0, scale=1.1))
        self.assertEqual(evaluate(mocap, path), evaluate(self.seq, path))

    def test_missing_ground_truth_exits_2(self):
        bare = os.path.join(self.dir, 'bare')
        synth.make_sequence(bare, N, gt_dir='other')
        with self.assertRaises(SystemExit) as e:
            evaluate(bare, self.tum(self.gt))
        self.assertEqual(e.exception.code, 2)

    def test_total_frames_without_cam1(self):
        mono = os.path.join(self.dir, 'mono')
        synth.make_sequence(mono, N, cam1=False)
        self.assertEqual(evaluate(mono, self.tum(self.gt))['total_frames'], N)

    def test_kornia_row_count_mismatch_exits_2(self):
        csv_path = os.path.join(self.dir, 'kornia_slam_raw.csv')
        log = os.path.join(self.dir, 'err.log')
        synth.write_kornia(csv_path, self.gt)
        synth.write_status_log(log, range(N - 1))
        with self.assertRaises(SystemExit) as e:
            evaluate(self.seq, csv_path, fmt='kornia', log=log)
        self.assertEqual(e.exception.code, 2)

    def test_kornia_rows_take_the_status_line_frames(self):
        start = 5
        csv_path = os.path.join(self.dir, 'kornia_slam_raw.csv')
        log = os.path.join(self.dir, 'err.log')
        synth.write_kornia(csv_path, synth.moved(self.gt[start:], R0, T0))
        synth.write_status_log(log, range(start, N))
        out = evaluate(self.seq, csv_path, fmt='kornia', log=log)
        self.assertEqual(out['n_assoc'], N - start)
        self.assertEqual(out['ate_se3_rmse'], 0.0)
        self.assertEqual(out['pose'], 'camera_centre')

    def test_kornia_needs_a_log(self):
        csv_path = os.path.join(self.dir, 'kornia_slam_raw.csv')
        synth.write_kornia(csv_path, self.gt)
        with self.assertRaises(SystemExit) as e:
            evaluate(self.seq, csv_path, fmt='kornia')
        self.assertEqual(e.exception.code, 2)

    def test_one_outlier_is_two_jumps(self):
        ps = [(r, p.copy()) for r, p in self.gt]
        ps[60][1][2] += 0.5
        self.assertEqual(evaluate(self.seq, self.tum(ps))['jumps'], 2)

    def test_permanent_step_is_one_jump(self):
        ps = [(r, p + (np.array([0, 0, 0.5]) if i >= 60 else 0)) for i, (r, p) in enumerate(self.gt)]
        self.assertEqual(evaluate(self.seq, self.tum(ps))['jumps'], 1)

    def test_keep_limits_to_matching_stamps(self):
        out = evaluate(self.seq, self.tum(self.gt), keep=self.ts[:40])
        self.assertEqual(out['n_est'], 40)
        self.assertEqual(out['n_assoc'], 40)


class ScoreTest(SequenceCase):
    def run_dir(self, name='main_seq_s0_r1', done=True, invalid=False):
        d = os.path.join(self.dir, 'runs', name)
        os.makedirs(d)
        with open(os.path.join(d, 'meta.txt'), 'w') as f:
            f.write(f'arm=main\nseq=seq\ndata={self.seq}\nstart=0\nrep=1\nbin_md5=0\npace=app\nload_start=0.1 0.2 0.3\n')
        synth.write_status_log(os.path.join(d, 'err.log'), range(N))
        synth.write_tum(os.path.join(d, 'traj.txt'), self.ts, self.gt)
        synth.write_kornia(os.path.join(d, 'kornia_slam_raw.csv'), self.gt)
        if done:
            write(os.path.join(d, 'done'), 'rc=0 wall=3 load_end=0.1 0.2 0.3\n')
        if invalid:
            write(os.path.join(d, 'invalid'), 'header mismatch\n')
        return d

    def test_not_done(self):
        self.assertEqual(score.one(self.run_dir(done=False))['status'], 'not done')

    def test_invalid(self):
        self.assertEqual(score.one(self.run_dir(done=False, invalid=True))['status'], 'invalid')

    def test_ok(self):
        r = score.one(self.run_dir())
        self.assertEqual(r['status'], 'ok')
        self.assertEqual((r['arm'], r['seq'], r['start'], r['rep'], r['rc']), ('main', 'seq', 0, 1, 0))
        self.assertEqual(r['traj.txt']['ate_se3_rmse'], 0.0)
        self.assertIn('rpe10', r['traj.txt'])
        self.assertEqual(r['kornia_slam_raw.csv']['n_assoc'], N)
        self.assertEqual(r['kornia_slam_raw.csv@nearest']['pose'], 'camera_centre')
        self.assertNotIn('traj_smoothed.txt', r)
        json.dumps(r)

    def test_refuses_a_foreign_file(self):
        d = self.run_dir()
        with self.assertRaises(SystemExit):
            score.ev(d, self.seq, 'kornia_slam_aligned.csv')

    def test_crop_to_keeps_matching_stamps(self):
        d = self.run_dir()
        ref = os.path.join(self.dir, 'ref', 'seq')
        os.makedirs(ref)
        synth.write_tum(os.path.join(ref, 'f_run.txt'), self.ts[10:50], self.gt[10:50])
        r = score.one(d, crop_to=os.path.join(self.dir, 'ref', '{seq}', 'f_run.txt'))
        self.assertEqual(r['traj.txt@crop']['n_est'], 40)
        self.assertEqual(r['kornia_slam_raw.csv@crop']['n_est'], 40)
        self.assertEqual(r['traj.txt']['n_est'], N)


if __name__ == '__main__':
    unittest.main()
