#!/usr/bin/env python3
"""tumvi_to_euroc.py SRC DST [--focal 190] [--size 512]: a TUM-VI sequence as a pinhole EuRoC sequence.

SRC is a dataset-*_512_16 export (mav0/ and dso/camchain.yaml). Each camera is undistorted from the
equidistant fisheye model to a pinhole of focal length --focal (px), principal point at the image
centre, same orientation, zero distortion: bilinear remap of the 16-bit image, then its high byte
(the app's own 16-to-8-bit reduction), written as 8-bit PNG. Each sensor.yaml carries
T_BS = T_cam_imu^-1 (body = IMU). imu0 is copied; mocap0 becomes state_groundtruth_estimate0, the
ground truth the EuRoC reader opens.
"""
import argparse
import os
import shutil

import cv2
import numpy as np
import yaml

CAMERAS = ('cam0', 'cam1')


def load_camchain(src):
    with open(os.path.join(src, 'dso', 'camchain.yaml')) as f:
        chain = yaml.safe_load(f)
    for cam in CAMERAS:
        if chain[cam]['distortion_model'] != 'equidistant':
            raise SystemExit(f'tumvi_to_euroc.py: {cam} is {chain[cam]["distortion_model"]}, expected equidistant')
    return chain


def pinhole(focal, size):
    c = (size - 1) / 2
    return np.array([[focal, 0, c], [0, focal, c], [0, 0, 1]], dtype=float)


def undistort_maps(calib, focal, size):
    """Per output pixel, the source pixel of the equidistant fisheye camera (x and y maps, float32)."""
    fx, fy, cx, cy = calib['intrinsics']
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    d = np.array(calib['distortion_coeffs'], dtype=float).reshape(4, 1)
    return cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), pinhole(focal, size), (size, size), cv2.CV_32FC1)


def convert_image(src_path, dst_path, maps):
    if not os.path.exists(src_path):
        raise SystemExit(f'tumvi_to_euroc.py: {src_path} is listed in data.csv but does not exist')
    wide = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if wide is None or wide.dtype != np.uint16 or wide.ndim != 2:
        raise SystemExit(f'tumvi_to_euroc.py: {src_path} is not a 16-bit grayscale image')
    out = cv2.remap(wide, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if not cv2.imwrite(dst_path, (out >> 8).astype(np.uint8)):
        raise SystemExit(f'tumvi_to_euroc.py: cannot write {dst_path}')


def sensor_yaml(calib, focal, size):
    c = (size - 1) / 2
    t_bs = np.linalg.inv(np.array(calib['T_cam_imu'], dtype=float))
    return {
        'camera_model': 'pinhole',
        'distortion_model': 'radial-tangential',
        'intrinsics': [float(focal), float(focal), c, c],
        'distortion_coefficients': [0.0, 0.0, 0.0, 0.0],
        'resolution': [size, size],
        'T_BS': {'rows': 4, 'cols': 4, 'data': [float(x) for x in t_bs.flatten()]},
    }


def image_names(csv_path):
    with open(csv_path) as f:
        return [line.split(',')[1].strip() for line in f if line.strip() and not line.startswith('#')]


def convert_camera(src, dst, cam, calib, focal, size):
    src_cam, dst_cam = os.path.join(src, 'mav0', cam), os.path.join(dst, 'mav0', cam)
    os.makedirs(os.path.join(dst_cam, 'data'), exist_ok=True)
    shutil.copyfile(os.path.join(src_cam, 'data.csv'), os.path.join(dst_cam, 'data.csv'))
    maps = undistort_maps(calib, focal, size)
    for name in image_names(os.path.join(src_cam, 'data.csv')):
        convert_image(os.path.join(src_cam, 'data', name), os.path.join(dst_cam, 'data', name), maps)
    with open(os.path.join(dst_cam, 'sensor.yaml'), 'w') as f:
        yaml.safe_dump(sensor_yaml(calib, focal, size), f, sort_keys=False, default_flow_style=None)


def copy_csv(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def convert(src, dst, focal=190.0, size=512):
    chain = load_camchain(src)
    for cam in CAMERAS:
        convert_camera(src, dst, cam, chain[cam], focal, size)
    copy_csv(os.path.join(src, 'mav0/imu0/data.csv'), os.path.join(dst, 'mav0/imu0/data.csv'))
    copy_csv(os.path.join(src, 'mav0/mocap0/data.csv'),
             os.path.join(dst, 'mav0/state_groundtruth_estimate0/data.csv'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--focal', type=float, default=190.0, help='pinhole focal length (px)')
    ap.add_argument('--size', type=int, default=512, help='output width and height (px)')
    a = ap.parse_args()
    convert(a.src, a.dst, a.focal, a.size)


if __name__ == '__main__':
    main()
