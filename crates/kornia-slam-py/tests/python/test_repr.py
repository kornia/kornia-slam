"""Repr / equality / shape sanity checks for the exposed types."""

from __future__ import annotations

import numpy as np
import pytest

from kornia_slam import PinholeCamera, TrackingStatus


def test_tracking_status_equality_and_repr() -> None:
    assert TrackingStatus.Tracked == TrackingStatus.Tracked
    assert TrackingStatus.Tracked != TrackingStatus.Skipped
    r = repr(TrackingStatus.Tracked)
    assert "Tracked" in r


def test_pinhole_camera_basic() -> None:
    cam = PinholeCamera(
        width=752,
        height=480,
        fx=458.654,
        fy=457.296,
        cx=367.215,
        cy=248.375,
        distortion=[-0.283, 0.074, 0.0, 0.0],
    )
    assert cam.width == 752
    assert cam.height == 480
    assert cam.fx == 458.654
    assert cam.distortion == [-0.283, 0.074, 0.0, 0.0]

    k = cam.intrinsic_matrix()
    assert k.shape == (3, 3)
    assert k.dtype == np.float64
    assert k[0, 0] == 458.654
    assert k[1, 1] == 457.296
    assert k[0, 2] == 367.215
    assert k[1, 2] == 248.375
    assert k[2, 2] == 1.0


def test_pinhole_camera_no_distortion() -> None:
    cam = PinholeCamera(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    assert cam.distortion == [0.0, 0.0, 0.0, 0.0]


def test_pinhole_camera_length5_distortion_with_zero_k3() -> None:
    cam = PinholeCamera(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        distortion=[-0.1, 0.02, 0.0, 0.0, 0.0],
    )
    assert cam.distortion == [-0.1, 0.02, 0.0, 0.0]


def test_pinhole_camera_rejects_nonzero_k3() -> None:
    with pytest.raises(ValueError):
        PinholeCamera(
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=[-0.1, 0.02, 0.0, 0.0, 0.0001],
        )


def test_pinhole_camera_rejects_wrong_distortion_length() -> None:
    with pytest.raises(ValueError):
        PinholeCamera(
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=[0.0, 0.0, 0.0],
        )


def test_pinhole_camera_rejects_zero_dim() -> None:
    with pytest.raises(ValueError):
        PinholeCamera(width=0, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
