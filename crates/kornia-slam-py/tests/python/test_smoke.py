"""End-to-end smoke test driven from Python.

Generates a synthetic camera motion + 3D points and renders frames the SLAM
pipeline can bootstrap on. The bar is low: we just want the binding to
accept input, run the pipeline, and return well-shaped results.
"""

from __future__ import annotations

import numpy as np
import pytest

from kornia_slam import OrbSlam, PinholeCamera, TrackingResult, TrackingStatus


def _make_camera() -> PinholeCamera:
    return PinholeCamera(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )


def _render_textured_frame(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    """Quick textured noise frame. Not real geometry — just enough corners that
    OrbDetector returns something. Slow drift between seeds keeps frames close
    enough for matching."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    # Burn in a couple of high-contrast squares so ORB has stable corners.
    cx, cy = 320 + (seed % 5), 240 + (seed % 5)
    for ox, oy in [(-100, -80), (-100, 80), (100, -80), (100, 80), (0, 0)]:
        x0, y0 = cx + ox - 10, cy + oy - 10
        img[y0 : y0 + 20, x0 : x0 + 20] = 255 if ((ox + oy) // 20) % 2 == 0 else 0
    return img


def test_constructs_with_camera() -> None:
    cam = _make_camera()
    slam = OrbSlam(camera=cam, n_keypoints=500)
    assert slam.num_active_map_points == 0
    assert slam.current_keyframe_idx is None
    pose = slam.current_pose
    assert pose.shape == (4, 4)
    assert pose.dtype == np.float64


def test_process_gray_returns_tracking_result() -> None:
    slam = OrbSlam(camera=_make_camera(), n_keypoints=500)
    img = _render_textured_frame(0)
    result = slam.process(img)
    assert isinstance(result, TrackingResult)
    assert result.frame_idx == 0
    assert result.pose.shape == (4, 4)
    assert result.pose.dtype == np.float64
    assert result.status in (
        TrackingStatus.Tracked,
        TrackingStatus.Skipped,
        TrackingStatus.KeyframeAccepted,
    )


def test_process_rgb_accepted() -> None:
    slam = OrbSlam(camera=_make_camera(), n_keypoints=500)
    gray = _render_textured_frame(0)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    result = slam.process(rgb)
    assert result.pose.shape == (4, 4)


def test_frame_idx_increments() -> None:
    slam = OrbSlam(camera=_make_camera(), n_keypoints=500)
    indices = []
    for seed in range(5):
        result = slam.process(_render_textured_frame(seed))
        indices.append(result.frame_idx)
    assert indices == [0, 1, 2, 3, 4]


def test_map_points_returns_n3_array() -> None:
    slam = OrbSlam(camera=_make_camera(), n_keypoints=500)
    # Before any frame: empty (0, 3) array.
    pts = slam.map_points()
    assert pts.shape == (0, 3)
    assert pts.dtype == np.float64


def test_rejects_wrong_dtype() -> None:
    slam = OrbSlam(camera=_make_camera(), n_keypoints=500)
    bad = np.zeros((480, 640), dtype=np.float32)
    with pytest.raises((ValueError, TypeError)):
        slam.process(bad)


def test_rejects_wrong_rank() -> None:
    slam = OrbSlam(camera=_make_camera(), n_keypoints=500)
    bad = np.zeros((480,), dtype=np.uint8)
    with pytest.raises((ValueError, TypeError)):
        slam.process(bad)


def test_rejects_wrong_channel_count() -> None:
    slam = OrbSlam(camera=_make_camera(), n_keypoints=500)
    bad = np.zeros((480, 640, 4), dtype=np.uint8)
    with pytest.raises((ValueError, TypeError)):
        slam.process(bad)
