import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import synth
from evaluate import evaluate
import score

SCORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'score.py')
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

    def test_total_frames_counts_cam0_frames_with_a_cam1_partner(self):
        cam1 = os.path.join(self.seq, 'mav0/cam1/data.csv')
        with open(cam1) as f:
            rows = f.read().splitlines()
        write(cam1, '\n'.join(rows[:1] + rows[4:]) + '\n')
        self.assertEqual(evaluate(self.seq, self.tum(self.gt))['total_frames'], N - 3)

    def test_explicit_zero_total_is_kept(self):
        out = evaluate(self.seq, self.tum(self.gt), total=0)
        self.assertEqual((out['total_frames'], out['tracked_pct']), (0, None))

    def test_nearest_gt_has_no_time_limit(self):
        late = os.path.join(self.dir, 'late')
        synth.make_sequence(late, N, gt_offset_ns=20_000_000)
        path = self.tum(self.gt)
        out = evaluate(late, path)
        self.assertEqual(out['n_assoc'], 0)
        self.assertIn('assoc_error', out)
        self.assertEqual(evaluate(late, path, nearest_gt=True)['n_assoc'], N)

    def test_tum_without_poses_exits_2(self):
        path = os.path.join(self.dir, 'traj.txt')
        write(path, '# t x y z qx qy qz qw\n')
        with self.assertRaises(SystemExit) as e:
            evaluate(self.seq, path)
        self.assertEqual(e.exception.code, 2)
        self.assertIn('no poses', e.exception.reason)

    def test_tum_short_row_exits_2(self):
        path = os.path.join(self.dir, 'traj.txt')
        write(path, f'{self.ts[0]} 0 0 0 0 0 0\n')
        with self.assertRaises(SystemExit) as e:
            evaluate(self.seq, path)
        self.assertIn('7 fields', e.exception.reason)

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

    def test_kornia_ignores_lines_that_are_not_status_lines(self):
        csv_path = os.path.join(self.dir, 'kornia_slam_raw.csv')
        log = os.path.join(self.dir, 'err.log')
        synth.write_kornia(csv_path, synth.moved(self.gt[:3], R0, T0))
        write(log, 'Dataset: synthetic\n[summary] wall=1\n[   12]x\n[    0] Tracked\n[pgo] failed\n'
                   '[    1] Tracked\n[    2] Lost\n')
        out = evaluate(self.seq, csv_path, fmt='kornia', log=log)
        self.assertEqual((out['n_assoc'], out['ate_se3_rmse']), (3, 0.0))

    def test_kornia_status_frame_past_cam0_exits_2(self):
        csv_path = os.path.join(self.dir, 'kornia_slam_raw.csv')
        log = os.path.join(self.dir, 'err.log')
        synth.write_kornia(csv_path, self.gt)
        synth.write_status_log(log, [*range(N - 1), N])
        with self.assertRaises(SystemExit) as e:
            evaluate(self.seq, csv_path, fmt='kornia', log=log)
        self.assertEqual(e.exception.code, 2)
        self.assertIn(f'names frame {N}', e.exception.reason)

    def test_kornia_without_rows_exits_2(self):
        csv_path = os.path.join(self.dir, 'kornia_slam_raw.csv')
        log = os.path.join(self.dir, 'err.log')
        synth.write_kornia(csv_path, [])
        synth.write_status_log(log, [])
        with self.assertRaises(SystemExit) as e:
            evaluate(self.seq, csv_path, fmt='kornia', log=log)
        self.assertIn('no poses', e.exception.reason)

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

    def test_keep_matching_nothing_exits_2(self):
        with self.assertRaises(SystemExit) as e:
            evaluate(self.seq, self.tum(self.gt), keep=[t + 1_000_000_000_000 for t in self.ts])
        self.assertIn('no pose within 5 ms', e.exception.reason)


class ScoreTest(SequenceCase):
    def run_dir(self, name='main_seq_s0_r1', done=True, invalid=False, seq=None, rc=0):
        d = os.path.join(self.dir, 'runs', name)
        os.makedirs(d)
        with open(os.path.join(d, 'meta.txt'), 'w') as f:
            f.write(f'arm=main\nseq=seq\ndata={seq or self.seq}\nstart=0\nrep=1\nbin_md5=0\npace=app\n'
                    'load_start=0.1 0.2 0.3\n')
        synth.write_status_log(os.path.join(d, 'err.log'), range(N))
        synth.write_tum(os.path.join(d, 'traj.txt'), self.ts, self.gt)
        synth.write_kornia(os.path.join(d, 'kornia_slam_raw.csv'), self.gt)
        if done:
            write(os.path.join(d, 'done'), f'rc={rc} wall=3 load_end=0.1 0.2 0.3\n')
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

    def test_nearest_key_uses_nearest_ground_truth(self):
        late = os.path.join(self.dir, 'late')
        synth.make_sequence(late, N, gt_offset_ns=20_000_000)
        r = score.one(self.run_dir(seq=late))
        self.assertEqual(r['kornia_slam_raw.csv']['n_assoc'], 0)
        self.assertEqual(r['kornia_slam_raw.csv@nearest']['n_assoc'], N)

    def test_nonzero_rc_is_failed(self):
        r = score.one(self.run_dir(rc=124))
        self.assertEqual((r['status'], r['rc']), ('failed', 124))
        self.assertEqual(r['traj.txt']['n_assoc'], N)

    def test_unscorable_file_gets_an_error_entry(self):
        d = self.run_dir()
        write(os.path.join(d, 'traj.txt'), '# header only\n')
        r = score.one(d)
        self.assertEqual(r['status'], 'ok')
        self.assertIn('no poses', r['traj.txt']['error'])
        self.assertEqual(r['kornia_slam_raw.csv']['n_assoc'], N)

    def test_malformed_directories_are_errors(self):
        no_meta = os.path.join(self.dir, 'runs', 'no_meta')
        os.makedirs(no_meta)
        self.assertEqual(score.one_or_error(no_meta)['status'], 'error')
        bad_done = self.run_dir('bad_done', done=False)
        write(os.path.join(bad_done, 'done'), 'killed\n')
        self.assertIn('rc=', score.one_or_error(bad_done)['error'])

    def test_main_scores_every_directory_after_an_error(self):
        bad = os.path.join(self.dir, 'runs', 'a_empty')
        os.makedirs(bad)
        good = self.run_dir('b_good')
        p = subprocess.run([sys.executable, SCORE, bad, good], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        rs = [json.loads(line) for line in p.stdout.splitlines()]
        self.assertEqual([r['status'] for r in rs], ['error', 'ok'])

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

    def test_crop_to_reads_seconds(self):
        d = self.run_dir()
        ref = os.path.join(self.dir, 'f_run.txt')
        synth.write_tum(ref, [f'{t * 1e-9:.6f}' for t in self.ts[10:50]], self.gt[10:50])
        r = score.one(d, crop_to=ref)
        self.assertEqual(r['traj.txt@crop']['n_est'], 40)


if __name__ == '__main__':
    unittest.main()
