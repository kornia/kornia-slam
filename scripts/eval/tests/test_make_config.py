import os
import re
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), 'make_config.sh')
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), 'configs')
KNOB = 'mapping: ( execution: Synchronous ),'


class MakeConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.seq = os.path.join(self.tmp.name, 'seq')
        os.makedirs(self.seq)
        self.out = os.path.join(self.tmp.name, 'run.ron')

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, preset, *knobs, vocab='/abs/ORBvoc.txt', seq=None):
        env = dict(os.environ, START_FRAME='7', MAX_FRAMES='50')
        env.pop('VOCAB', None)
        if vocab:
            env['VOCAB'] = vocab
        return subprocess.run(['bash', SCRIPT, os.path.join(CONFIGS, preset), seq or self.seq, self.out, *knobs],
                              env=env, capture_output=True, text=True)

    def raw(self):
        with open(self.out) as f:
            return f.read()

    def preset(self, text):
        path = os.path.join(self.tmp.name, 'preset.ron')
        with open(path, 'w') as f:
            f.write(text)
        return path

    def code(self):
        with open(self.out) as f:
            return [line.split('//')[0] for line in f]

    def check(self, preset, knobs):
        r = self.make(preset, *([KNOB] if knobs else []))
        self.assertEqual(r.returncode, 0, r.stderr)
        code = self.code()
        text = ''.join(code)
        self.assertEqual(text.count(f'data: "{self.seq}"'), 1)
        self.assertEqual(len(re.findall(r'\bstart_frame: 7\b', text)), 1)
        self.assertEqual(len(re.findall(r'\bstart_frame:', text)), 1)
        self.assertEqual(len(re.findall(r'\bmax_frames: 50\b', text)), 1)
        self.assertEqual(len(re.findall(r'\bmax_frames:', text)), 1)
        self.assertNotIn('ORBvoc.txt"', text.replace('"/abs/ORBvoc.txt"', ''))
        if knobs:
            i = next(i for i, line in enumerate(code) if 'pipeline: OrbSlam((' in line)
            self.assertEqual(code[i + 1].strip(), KNOB)
        return text

    def test_euroc(self):
        text = self.check('euroc.ron', knobs=True)
        self.assertIn('max_frames_between: 8', text)

    def test_euroc_stereo_imu_loop(self):
        text = self.check('euroc-stereo-imu-loop.ron', knobs=True)
        self.assertIn('vocabulary: "/abs/ORBvoc.txt"', text)

    def test_tumvi_imu(self):
        self.check('tumvi-imu.ron', knobs=False)

    def test_knob_without_pipeline_exits_2(self):
        r = self.make('tumvi-imu.ron', KNOB)
        self.assertEqual(r.returncode, 2)
        self.assertFalse(os.path.exists(self.out))

    def test_vocabulary_preset_without_vocab_exits_2(self):
        self.assertEqual(self.make('euroc-stereo-imu-loop.ron', vocab=None).returncode, 2)

    def test_vocabulary_in_a_comment_needs_no_vocab(self):
        r = self.make('euroc.ron', vocab=None)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(CONFIGS, 'euroc.ron')) as f:
            comment = next(line for line in f if 'vocabulary:' in line)
        self.assertIn(comment, self.raw())

    def test_values_are_copied_verbatim(self):
        seq = os.path.join(self.tmp.name, 'se&q\\n')
        os.makedirs(seq)
        knob = 'name: "a\\nb & \\"c\\"",'
        preset = self.preset('(\n    source: Euroc(( data: "x" )),\n    pipeline: OrbSlam((\n    )),\n)\n')
        r = self.make(preset, knob, seq=seq)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = self.raw()
        self.assertIn(f'data: "{seq}", start_frame: 7, max_frames: 50', text)
        self.assertIn('            ' + knob + '\n', text)

    def test_only_exact_frame_keys_are_dropped(self):
        preset = self.preset('( source: Euroc(( data: "x", start_frame: 2, max_frames: 9 )),\n'
                             '  other: ( my_max_frames: 3, max_frames_between: 8, xstart_frame: 4 ) )\n')
        r = self.make(preset)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = self.raw()
        self.assertIn('my_max_frames: 3, max_frames_between: 8, xstart_frame: 4', text)
        self.assertEqual(len(re.findall(r'(?<![A-Za-z0-9_])max_frames:', text)), 1)
        self.assertNotIn('start_frame: 2', text)

    def test_two_data_paths_exit_3(self):
        preset = os.path.join(self.tmp.name, 'two.ron')
        with open(preset, 'w') as f:
            f.write('( source: Euroc(( data: "a" )), other: ( data: "b" ) )\n')
        self.assertEqual(self.make(preset).returncode, 3)


if __name__ == '__main__':
    unittest.main()
