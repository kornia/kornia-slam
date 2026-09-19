//! Culling unreliable map points.

use crate::map::Map;

impl Map {
    /// Cull map points with poor observation ratios or that project behind cameras.
    ///
    /// Removal goes through [`Map::remove_landmark`], so each culled landmark
    /// clears exactly the features its own observation records name. The
    /// previous sweep over every keyframe association produced the same result
    /// only while the two stayed in step; going through the canonical path
    /// makes that structural.
    pub fn cull(&mut self) {
        const MIN_OBSERVATIONS: u32 = 5;
        const MIN_FOUND_RATIO: f64 = 0.20;

        let mut doomed: Vec<usize> = Vec::new();

        for (idx, mp) in self.map_points.iter().enumerate() {
            if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
                continue;
            }
            if mp.found_ratio() < MIN_FOUND_RATIO {
                doomed.push(idx);
            }
        }

        for kf in &self.keyframes {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    let p_cam = kf.frame.pose_world_to_cam.transform_point(&mp.position);
                    if p_cam.z <= 1e-8 {
                        doomed.push(*mp_idx);
                    }
                }
            }
        }

        for idx in doomed {
            let _ = self.remove_landmark(idx);
        }
    }
}
