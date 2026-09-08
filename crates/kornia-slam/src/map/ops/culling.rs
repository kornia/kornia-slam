//! Redundant map-point culling. Deletion is logical: indices stay valid and
//! the point vector is never compacted.

use crate::map::Map;
use std::collections::HashSet;

impl Map {
    /// Cull map points with poor observation ratios or that project behind cameras.
    pub fn cull(&mut self) {
        const MIN_OBSERVATIONS: u32 = 5;
        const MIN_FOUND_RATIO: f64 = 0.20;

        let mut n_culled = 0usize;

        for mp in self.map_points.iter_mut() {
            if mp.culled || mp.n_visible < MIN_OBSERVATIONS {
                continue;
            }
            if mp.found_ratio() < MIN_FOUND_RATIO {
                mp.mark_culled();
                n_culled += 1;
            }
        }

        let mut behind_camera: Vec<usize> = Vec::new();
        for kf in &self.keyframes {
            for mp_idx in kf.map_point_by_desc_idx.iter().flatten() {
                if let Some(mp) = self.map_points.get(*mp_idx)
                    && !mp.culled
                {
                    let p_cam = kf.frame.pose_world_to_cam.transform_point(&mp.position);
                    if p_cam.z <= 1e-8 {
                        behind_camera.push(*mp_idx);
                    }
                }
            }
        }

        for mp_idx in &behind_camera {
            if let Some(mp) = self.map_points.get_mut(*mp_idx)
                && !mp.culled
            {
                mp.mark_culled();
                n_culled += 1;
            }
        }

        if n_culled > 0 {
            let culled_set: HashSet<usize> = self
                .map_points
                .iter()
                .enumerate()
                .filter(|(_, mp)| mp.culled)
                .map(|(i, _)| i)
                .collect();

            for kf in &mut self.keyframes {
                for desc_idx in 0..kf.map_point_by_desc_idx.len() {
                    if let Some(mp_idx) = kf.map_point(desc_idx)
                        && culled_set.contains(&mp_idx)
                    {
                        kf.clear_map_point(desc_idx);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::map::{Map, MapPoint};
    use kornia_algebra::Vec3F64;

    #[test]
    fn cull_map_points_removes_low_ratio() {
        let mut map = Map::new();

        let first_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(0.0, 0.0, 5.0),
            [0u8; 32],
            0,
            [0; 3],
            0,
        ));
        let second_idx = map.push_map_point(MapPoint::new(
            Vec3F64::new(1.0, 0.0, 5.0),
            [1u8; 32],
            0,
            [0; 3],
            0,
        ));
        map.map_points_mut()[first_idx].n_visible = 10;
        map.map_points_mut()[first_idx].n_found = 1;
        map.map_points_mut()[second_idx].n_visible = 10;
        map.map_points_mut()[second_idx].n_found = 5;

        map.cull();

        assert!(map.map_points()[first_idx].culled);
        assert!(!map.map_points()[second_idx].culled);
    }
}
