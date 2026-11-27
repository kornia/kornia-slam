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

    def make(self, preset, *knobs, vocab='/abs/ORBvoc.txt'):
        env = dict(os.environ, START_FRAME='7', MAX_FRAMES='50')
        env.pop('VOCAB', None)
        if vocab:
            env['VOCAB'] = vocab
        return subprocess.run(['bash', SCRIPT, os.path.join(CONFIGS, preset), self.seq, self.out, *knobs],
                              env=env, capture_output=True, text=True)

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

    def test_two_data_paths_exit_3(self):
        preset = os.path.join(self.tmp.name, 'two.ron')
        with open(preset, 'w') as f:
            f.write('( source: Euroc(( data: "a" )), other: ( data: "b" ) )\n')
        self.assertEqual(self.make(preset).returncode, 3)


if __name__ == '__main__':
    unittest.main()
