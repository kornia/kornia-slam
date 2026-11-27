import csv
import os
import tempfile
import unittest

import cv2
import numpy as np
import yaml

import synth  # noqa: F401  (puts scripts/eval on sys.path)
import tumvi_to_euroc as conv

# cam0 of TUM-VI's published Kalibr chain.
CALIB = {
    'T_cam_imu': [[-0.9995250378696743, 0.029615343885863205, -0.008522328211654736, 0.04727988224914392],
                  [0.0075019185074052044, -0.03439736061393144, -0.9993800792498829, -0.047443232143367084],
                  [-0.02989013031643309, -0.998969345370175, 0.03415885127385616, -0.0681999605066297],
                  [0.0, 0.0, 0.0, 1.0]],
    'camera_model': 'pinhole',
    'distortion_coeffs': [0.0034823894022493434, 0.0007150348452162257, -0.0020532361418706202,
                          0.00020293673591811182],
    'distortion_model': 'equidistant',
    'intrinsics': [190.97847715128717, 190.9733070521226, 254.93170605935475, 256.8974428996504],
    'resolution': [512, 512],
}
SENSOR_KEYS = ['camera_model', 'distortion_model', 'intrinsics', 'distortion_coefficients', 'resolution', 'T_BS']


def equidistant(calib, focal, size, u, v):
    """Closed-form source pixel of output pinhole pixel (u, v)."""
    c = (size - 1) / 2
    x, y = (u - c) / focal, (v - c) / focal
    r = np.hypot(x, y)
    fx, fy, cx, cy = calib['intrinsics']
    if r == 0:
        return cx, cy
    k1, k2, k3, k4 = calib['distortion_coeffs']
    th = np.arctan(r)
    th_d = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
    return fx * th_d / r * x + cx, fy * th_d / r * y + cy


class PixelTest(unittest.TestCase):
    """Remap on the 16-bit image, bilinear, then the high byte."""

    def convert(self, img):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, 'a.png'), os.path.join(tmp, 'b.png')
            cv2.imwrite(src, img)
            calib = dict(CALIB, intrinsics=[24.0, 24.0, 31.5, 31.5])
            maps = conv.undistort_maps(calib, 24.0, 64)
            conv.convert_image(src, dst, maps)
            return cv2.imread(dst, cv2.IMREAD_UNCHANGED), maps

    def test_high_byte_of_a_constant_image(self):
        out, _ = self.convert(np.full((64, 64), 0x1234, dtype=np.uint16))
        self.assertEqual(out[32, 32], 0x12)

    def test_bilinear_on_a_ramp(self):
        # 1024 per source column: the 8-bit output is 4 x (source x), floored; nearest would be up to 2 off.
        ramp = np.tile(np.arange(64, dtype=np.uint16) * 1024, (64, 1))
        out, (mx, _) = self.convert(ramp)
        frac = mx[8:56, 8:56] - np.floor(mx[8:56, 8:56])
        v, u = np.argwhere((frac > 0.4) & (frac < 0.6))[0] + 8
        self.assertLess(abs(float(out[v, u]) - 4 * float(mx[v, u])), 1.1)

    def test_missing_source_image_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as e:
                conv.convert_image(os.path.join(tmp, 'gone.png'), os.path.join(tmp, 'b.png'), None)
            self.assertIn('does not exist', str(e.exception))

    def test_unwritable_destination_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'a.png')
            cv2.imwrite(src, np.zeros((64, 64), dtype=np.uint16))
            maps = conv.undistort_maps(dict(CALIB, intrinsics=[24.0, 24.0, 31.5, 31.5]), 24.0, 64)
            with self.assertRaises(SystemExit) as e:
                conv.convert_image(src, os.path.join(tmp, 'no', 'dir', 'b.png'), maps)
            self.assertIn('cannot write', str(e.exception))


class MapTest(unittest.TestCase):
    def test_map_matches_closed_form_equidistant(self):
        # An odd size puts the principal point (size - 1) / 2 on a pixel.
        focal, size = 190.0, 511
        mx, my = conv.undistort_maps(CALIB, focal, size)
        for u, v in [(255, 255), (400, 255), (100, 60), (480, 470)]:
            ex, ey = equidistant(CALIB, focal, size, u, v)
            self.assertAlmostEqual(float(mx[v, u]), ex, delta=1e-3)
            self.assertAlmostEqual(float(my[v, u]), ey, delta=1e-3)


class ConvertTest(unittest.TestCase):
    SIZE, FOCAL, N = 64, 24.0, 2

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, 'dataset-room1_512_16')
        self.dst = os.path.join(self.tmp.name, 'room1')
        calib = dict(CALIB, intrinsics=[24.0, 24.0, 31.5, 31.5], resolution=[64, 64])
        os.makedirs(os.path.join(self.src, 'dso'))
        with open(os.path.join(self.src, 'dso', 'camchain.yaml'), 'w') as f:
            yaml.safe_dump({'cam0': calib, 'cam1': calib}, f)
        self.ts = synth.stamps(self.N)
        rng = np.random.default_rng(0)
        for cam in conv.CAMERAS:
            os.makedirs(os.path.join(self.src, 'mav0', cam, 'data'))
            synth.write_csv(os.path.join(self.src, 'mav0', cam, 'data.csv'), '#timestamp [ns],filename',
                            [[t, f'{t}.png'] for t in self.ts])
            for t in self.ts:
                img = rng.integers(0, 65535, (64, 64), dtype=np.uint16)
                cv2.imwrite(os.path.join(self.src, 'mav0', cam, 'data', f'{t}.png'), img)
        synth.write_csv(os.path.join(self.src, 'mav0/imu0/data.csv'), '#timestamp,w_x,w_y,w_z,a_x,a_y,a_z',
                        [[t, 0, 0, 0, 0, 0, 9.81] for t in self.ts])
        synth.write_csv(os.path.join(self.src, 'mav0/mocap0/data.csv'), '#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z',
                        [[t, 0, 0, 0, 1, 0, 0, 0] for t in self.ts])
        conv.convert(self.src, self.dst, self.FOCAL, self.SIZE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_images_are_8_bit_and_sized(self):
        for cam in conv.CAMERAS:
            for t in self.ts:
                img = cv2.imread(os.path.join(self.dst, 'mav0', cam, 'data', f'{t}.png'), cv2.IMREAD_UNCHANGED)
                self.assertEqual(img.dtype, np.uint8)
                self.assertEqual(img.shape, (self.SIZE, self.SIZE))

    def test_data_csv_rows_preserved(self):
        for cam in conv.CAMERAS:
            with open(os.path.join(self.src, 'mav0', cam, 'data.csv')) as a, \
                    open(os.path.join(self.dst, 'mav0', cam, 'data.csv')) as b:
                self.assertEqual(a.read(), b.read())

    def test_sensor_yaml(self):
        for cam in conv.CAMERAS:
            with open(os.path.join(self.dst, 'mav0', cam, 'sensor.yaml')) as f:
                s = yaml.safe_load(f)
            self.assertEqual(list(s), SENSOR_KEYS)
            self.assertEqual(s['camera_model'], 'pinhole')
            self.assertEqual(s['distortion_model'], 'radial-tangential')
            self.assertEqual(s['intrinsics'], [24.0, 24.0, 31.5, 31.5])
            self.assertEqual(s['distortion_coefficients'], [0.0] * 4)
            self.assertEqual(s['resolution'], [64, 64])
            t_bs = np.array(s['T_BS']['data']).reshape(4, 4)
            np.testing.assert_allclose(t_bs @ np.array(CALIB['T_cam_imu']), np.eye(4), atol=1e-12)

    def test_imu_and_ground_truth(self):
        self.assertTrue(os.path.exists(os.path.join(self.dst, 'mav0/imu0/data.csv')))
        with open(os.path.join(self.dst, 'mav0/state_groundtruth_estimate0/data.csv')) as f:
            rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
        self.assertEqual(len(rows), self.N)
        self.assertTrue(all(len(r) >= 8 for r in rows))


if __name__ == '__main__':
    unittest.main()
