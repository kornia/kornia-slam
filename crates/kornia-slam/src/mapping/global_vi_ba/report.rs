//! The `[offline_ba]` lines: every counter card D's predictions read.

use std::fmt;

use super::{CalibrationEstimate, OfflineBaReport};

impl OfflineBaReport {
    /// Fills the calibration fields in the units the card predicts them in.
    pub(super) fn record_calibration(&mut self, estimate: &CalibrationEstimate) {
        let norm_deg = |theta: &[f64; 3]| {
            (theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2])
                .sqrt()
                .to_degrees()
        };
        self.camera_rotation_deg = estimate
            .camera_rotation
            .iter()
            .map(|slices| slices.iter().map(norm_deg).collect())
            .collect();
        self.camera_focal_pct = estimate
            .camera_focal
            .iter()
            .map(|slices| slices.iter().map(|s| s * 100.0).collect())
            .collect();
        if let Some(gyro) = estimate.gyro_intrinsics {
            self.gyro_scale_pct = Some(gyro.scale.to_array().map(|s| s * 100.0));
            self.gyro_shear_mrad = Some(gyro.shear.to_array().map(|s| s * 1e3));
        }
        self.td_ms = estimate.clock_offset_sec.map(|td| td * 1e3);
    }
}

fn joined(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(",")
}

impl fmt::Display for OfflineBaReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // dropped_10px / dropped_3px name the two outlier stages as the card does.
        writeln!(
            f,
            "[offline_ba] frames={} iterations={} cost0={:.6e} cost={:.6e} dropped_10px={} dropped_3px={}",
            self.frames,
            self.iterations,
            self.initial_cost,
            self.final_cost,
            self.dropped_outlier,
            self.dropped_tight_outlier
        )?;
        writeln!(
            f,
            "[offline_ba] covis_median={} nnz_m={} nnz_l={}",
            self.covisible_median, self.nnz_reduced, self.nnz_factor
        )?;
        writeln!(f, "[offline_ba] iter_ms={}", joined(&self.iteration_ms))?;
        writeln!(
            f,
            "[offline_ba] components={} bridged={} gap_max_s={:.3} bridge_rot_whitened={:.4}",
            self.components, self.bridged, self.gap_max_s, self.bridge_rot_whitened
        )?;
        writeln!(
            f,
            "[offline_ba] imu_rot_whitened0={:.4} imu_rot_whitened={:.4}",
            self.imu_rot_whitened0, self.imu_rot_whitened
        )?;
        for (camera, slices) in self.camera_rotation_deg.iter().enumerate() {
            for (slice, deg) in slices.iter().enumerate() {
                writeln!(f, "[offline_ba] cam{camera} slice{slice} rot_deg={deg:.4}")?;
            }
        }
        for (camera, slices) in self.camera_focal_pct.iter().enumerate() {
            for (slice, pct) in slices.iter().enumerate() {
                writeln!(
                    f,
                    "[offline_ba] cam{camera} slice{slice} focal_pct={pct:.4}"
                )?;
            }
        }
        if let (Some(scale), Some(shear)) = (self.gyro_scale_pct, self.gyro_shear_mrad) {
            writeln!(
                f,
                "[offline_ba] gyro_scale_pct={} gyro_shear_mrad={}",
                joined(&scale),
                joined(&shear)
            )?;
        }
        if let (Some(td), Some(total)) = (self.td_ms, self.td_total_ms) {
            writeln!(f, "[offline_ba] td_ms={td:.4} td_total_ms={total:.4}")?;
        }
        Ok(())
    }
}
