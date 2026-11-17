#!/usr/bin/env python3
"""Unit tests for LaMAria adapter and start-time evaluation harness."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from lamaria_adapter import (
    LamariaAdapter,
    adjust_imu_csv,
    convert_pseudo_gt_to_euroc,
    create_lamaria_fixture,
    parse_lamaria_calibration,
    quat_to_rot_matrix,
)
from start_time_eval import (
    count_dataset_frames,
    generate_override_ron,
    parse_offsets,
    run_synthetic_smoke_test,
    sanitize_json,
)


class TestLamariaAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="lamaria_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_quat_to_rot_matrix_identity(self) -> None:
        r = quat_to_rot_matrix(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(r[0][0], 1.0)
        self.assertAlmostEqual(r[1][1], 1.0)
        self.assertAlmostEqual(r[2][2], 1.0)
        self.assertAlmostEqual(r[0][1], 0.0)

    def test_quat_to_rot_matrix_orthonormal(self) -> None:
        # 90 degrees around Z axis: qw=cos(45)=sqrt(0.5), qz=sin(45)=sqrt(0.5)
        s = math.sqrt(0.5)
        r = quat_to_rot_matrix(s, 0.0, 0.0, s)
        self.assertAlmostEqual(r[0][0], 0.0)
        self.assertAlmostEqual(r[0][1], -1.0)
        self.assertAlmostEqual(r[1][0], 1.0)
        self.assertAlmostEqual(r[1][1], 0.0)
        self.assertAlmostEqual(r[2][2], 1.0)
        # Determinant == 1.0
        det = (
            r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
        )
        self.assertAlmostEqual(det, 1.0)

    def test_fixture_creation_and_adaptation(self) -> None:
        raw_fixture = self.temp_dir / "raw_fixture"
        create_lamaria_fixture(raw_fixture, n_frames=5, fps=20.0, width=64, height=48)

        self.assertTrue((raw_fixture / "cam0" / "data.csv").exists())
        self.assertTrue((raw_fixture / "imu0" / "data.csv").exists())
        self.assertEqual(count_dataset_frames(raw_fixture), 5)

        # Adapt sequence
        adapted_dir = self.temp_dir / "adapted"
        adapter = LamariaAdapter(raw_fixture, imu_time_offset_ns=4_200_000)
        adapter.adapt(adapted_dir)

        mav0 = adapted_dir / "mav0"
        self.assertTrue((mav0 / "cam0" / "data.csv").exists())
        self.assertTrue((mav0 / "cam0" / "sensor.yaml").exists())
        self.assertTrue((mav0 / "imu0" / "data.csv").exists())
        self.assertTrue((mav0 / "state_groundtruth_estimate0" / "data.csv").exists())

        # Check IMU timestamp adjustment
        with open(raw_fixture / "imu0" / "data.csv", "r", encoding="utf-8") as f_raw:
            raw_lines = [l.strip() for l in f_raw if l.strip() and not l.startswith("#")]
        with open(mav0 / "imu0" / "data.csv", "r", encoding="utf-8") as f_adj:
            adj_lines = [l.strip() for l in f_adj if l.strip() and not l.startswith("#")]

        raw_ts = int(raw_lines[0].split(",")[0])
        adj_ts = int(adj_lines[0].split(",")[0])
        self.assertEqual(raw_ts - adj_ts, 4_200_000)

    def test_override_ron_generation_validation(self) -> None:
        base_ron = self.temp_dir / "base.ron"
        with open(base_ron, "w", encoding="utf-8") as f:
            f.write("""(
    source: Euroc((
        data: "old/path",
        start_frame: 0,
    )),
    system: (
        mapping: ( execution: Asynchronous ),
    ),
)""")

        out_ron = self.temp_dir / "out.ron"
        dataset_path = Path("new/path")
        generate_override_ron(base_ron, dataset_path, 50, out_ron)

        with open(out_ron, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn(str(dataset_path.resolve()).replace("\\", "/"), content)
        self.assertIn("start_frame: 50", content)
        self.assertIn("execution: Synchronous", content)
        self.assertNotIn("execution: Asynchronous", content)

    def test_sanitize_json(self) -> None:
        obj = {
            "a": float("nan"),
            "b": float("inf"),
            "c": 1.23,
            "d": [float("nan"), {"e": float("-inf")}],
        }
        sanitized = sanitize_json(obj)
        self.assertIsNone(sanitized["a"])
        self.assertIsNone(sanitized["b"])
        self.assertEqual(sanitized["c"], 1.23)
        self.assertIsNone(sanitized["d"][0])
        self.assertIsNone(sanitized["d"][1]["e"])

        # Must serialize with allow_nan=False
        dumped = json.dumps(sanitized, allow_nan=False)
        self.assertIn("null", dumped)


if __name__ == "__main__":
    unittest.main()
