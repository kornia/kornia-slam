import os
import stat
import subprocess
import tempfile
import unittest

import synth  # noqa: F401  (puts scripts/eval on sys.path)
import score

EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(EVAL, 'run.sh')

# Stands in for the app: lists flags on --help, records its arguments, prints a header then `Dataset:`.
STUB = '''#!/bin/bash
if [ "$1" = --help ]; then echo "--config --evaluate --eval-out --no-tui --debug ${STUB_HELP:-}"; exit 0; fi
echo "$*" >> "$STUB_CALLS"
if [ "${STUB_DIE:-0}" = 1 ]; then echo "Error: bad config" >&2; exit 3; fi
echo "system: ${STUB_HEADER:-a}" >&2
echo "Dataset: 10 frames (processing 0..10)" >&2
exit "${STUB_RC:-0}"
'''
PRESET = '(\n    source: Euroc(( data: "x" )),\n    pipeline: OrbSlam((\n    )),\n)\n'


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.bin = os.path.join(t, 'kornia-slam')
        with open(self.bin, 'w') as f:
            f.write(STUB)
        os.chmod(self.bin, os.stat(self.bin).st_mode | stat.S_IXUSR)
        self.preset = os.path.join(t, 'preset.ron')
        with open(self.preset, 'w') as f:
            f.write(PRESET)
        self.calls_path = os.path.join(t, 'calls')
        self.out = os.path.join(t, 'runs')
        self.seqs = [os.path.join(t, s) for s in ('MH_01', 'V1_02')]
        for s in self.seqs:
            os.makedirs(s)

    def tearDown(self):
        self.tmp.cleanup()

    def run_sh(self, seqs=None, preset=None, **env):
        e = dict(os.environ, BIN=self.bin, STUB_CALLS=self.calls_path)
        for k in ('VOCAB', 'KNOBS', 'PACE', 'RUNS', 'JOBS', 'STUB_HELP', 'STUB_DIE', 'STUB_HEADER', 'STUB_RC'):
            e.pop(k, None)
        e.update(env)
        return subprocess.run(['bash', RUN, 'main', preset or self.preset, self.out, *(seqs or self.seqs[:1])],
                              env=e, capture_output=True, text=True)

    def run_dir(self, seq='MH_01'):
        return os.path.join(self.out, f'main_{seq}_s0_r1')

    def read(self, path):
        with open(path) as f:
            return f.read()

    def calls(self):
        return self.read(self.calls_path).splitlines() if os.path.exists(self.calls_path) else []

    def test_ok_run_writes_meta_and_done_then_is_skipped(self):
        p = self.run_sh()
        self.assertEqual(p.returncode, 0, p.stderr)
        d = self.run_dir()
        self.assertTrue(self.read(os.path.join(d, 'done')).startswith('rc=0 '))
        meta = dict(line.split('=', 1) for line in self.read(os.path.join(d, 'meta.txt')).splitlines())
        self.assertEqual(sorted(meta), ['arm', 'bin_md5', 'data', 'load_start', 'pace', 'rep', 'seq', 'start'])
        self.assertEqual((meta['arm'], meta['seq'], meta['data'], meta['pace']), ('main', 'MH_01', self.seqs[0], 'app'))
        self.assertEqual(self.read(os.path.join(self.out, 'main.header')), 'system: a\n')
        self.assertEqual(self.run_sh().returncode, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_header_mismatch_is_invalid_and_exits_1(self):
        self.assertEqual(self.run_sh().returncode, 0)
        p = self.run_sh(self.seqs[1:], STUB_HEADER='b')
        self.assertEqual(p.returncode, 1)
        d = self.run_dir('V1_02')
        self.assertTrue(os.path.exists(os.path.join(d, 'invalid')))
        self.assertFalse(os.path.exists(os.path.join(d, 'done')))

    def test_run_that_dies_before_dataset_neither_seeds_nor_trips_the_guard(self):
        p = self.run_sh(STUB_DIE='1')
        self.assertEqual(p.returncode, 1)
        self.assertTrue(self.read(os.path.join(self.run_dir(), 'done')).startswith('rc=3 '))
        self.assertFalse(os.path.exists(os.path.join(self.out, 'main.header')))
        self.assertEqual(self.run_sh(self.seqs).returncode, 0)
        self.assertEqual(len(self.calls()), 3)
        for seq in ('MH_01', 'V1_02'):
            self.assertTrue(self.read(os.path.join(self.run_dir(seq), 'done')).startswith('rc=0 '))

    def test_nonzero_rc_after_dataset_does_not_seed_the_header(self):
        self.assertEqual(self.run_sh(STUB_RC='124', STUB_HEADER='partial').returncode, 1)
        self.assertFalse(os.path.exists(os.path.join(self.out, 'main.header')))

    def test_trajectory_out_only_when_help_lists_it(self):
        self.run_sh()
        self.run_sh(self.seqs[1:], STUB_HELP='--trajectory-out')
        first, second = self.calls()
        self.assertNotIn('--trajectory-out', first)
        self.assertIn(f'--trajectory-out {self.run_dir("V1_02")}/traj.txt', second)

    def test_pace_flag_must_be_listed(self):
        self.assertEqual(self.run_sh(PACE='none').returncode, 2)
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.run_sh(PACE='none', STUB_HELP='--no-pace').returncode, 0)
        self.assertIn('--no-pace', self.calls()[0])

    def test_missing_binary_exits_2(self):
        p = self.run_sh(BIN=os.path.join(self.tmp.name, 'nope'))
        self.assertEqual(p.returncode, 2)
        self.assertIn('not an executable file', p.stderr)

    def test_config_failure_exits_1_and_scores_not_done(self):
        preset = os.path.join(self.tmp.name, 'vocab.ron')
        with open(preset, 'w') as f:
            f.write(PRESET.replace('OrbSlam((', 'OrbSlam(( loop_closing: Enabled(vocabulary: "v"),'))
        p = self.run_sh(preset=preset)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(self.calls(), [])
        self.assertEqual(score.one(self.run_dir())['status'], 'not done')


if __name__ == '__main__':
    unittest.main()
