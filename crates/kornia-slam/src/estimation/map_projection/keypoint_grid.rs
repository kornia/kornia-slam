//! Spatial grid that mirrors ORB-SLAM3 `Frame::GetFeaturesInArea`.

const FRAME_GRID_COLS: usize = 64;
const FRAME_GRID_ROWS: usize = 48;

/// A 2D grid over undistorted image bounds for ORB-SLAM3-style keypoint lookups.
pub(super) struct KeypointGrid {
    cells: Vec<Vec<usize>>,
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    grid_w_inv: f32,
    grid_h_inv: f32,
}

impl KeypointGrid {
    /// Builds the grid using ORB-SLAM3's fixed 64x48 layout over undistorted
    /// image bounds. Keypoints outside the bounds are skipped.
    pub(super) fn new(
        keypoints_xy: &[[f32; 2]],
        image_bounds: (f32, f32, f32, f32),
    ) -> Self {
        let (min_x, max_x, min_y, max_y) = image_bounds;
        let width = (max_x - min_x).max(1.0);
        let height = (max_y - min_y).max(1.0);
        let grid_w_inv = FRAME_GRID_COLS as f32 / width;
        let grid_h_inv = FRAME_GRID_ROWS as f32 / height;
        let mut cells = vec![Vec::new(); FRAME_GRID_COLS * FRAME_GRID_ROWS];

        for (i, kp) in keypoints_xy.iter().enumerate() {
            if let Some((col, row)) = pos_in_grid(kp[0], kp[1], min_x, min_y, grid_w_inv, grid_h_inv)
            {
                cells[row * FRAME_GRID_COLS + col].push(i);
            }
        }

        Self {
            cells,
            min_x,
            max_x,
            min_y,
            max_y,
            grid_w_inv,
            grid_h_inv,
        }
    }

    /// Returns indices matching ORB-SLAM3 `GetFeaturesInArea`: square search
    /// window, optional octave range, and undistorted-bounds grid traversal.
    pub(super) fn query_features_in_area(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        min_level: isize,
        max_level: isize,
        keypoints_xy: &[[f32; 2]],
        scales: &[f32],
        keypoint_octave: impl Fn(f32) -> usize,
    ) -> Vec<usize> {
        let mut result = Vec::new();

        let n_min_cell_x = (((x - self.min_x - radius) * self.grid_w_inv).floor() as isize)
            .max(0);
        if n_min_cell_x >= FRAME_GRID_COLS as isize {
            return result;
        }
        let n_max_cell_x = (((x - self.min_x + radius) * self.grid_w_inv).ceil() as isize)
            .min(FRAME_GRID_COLS as isize - 1);
        if n_max_cell_x < 0 {
            return result;
        }

        let n_min_cell_y = (((y - self.min_y - radius) * self.grid_h_inv).floor() as isize)
            .max(0);
        if n_min_cell_y >= FRAME_GRID_ROWS as isize {
            return result;
        }
        let n_max_cell_y = (((y - self.min_y + radius) * self.grid_h_inv).ceil() as isize)
            .min(FRAME_GRID_ROWS as isize - 1);
        if n_max_cell_y < 0 {
            return result;
        }

        let check_levels = min_level > 0 || max_level >= 0;
        for ix in n_min_cell_x..=n_max_cell_x {
            for iy in n_min_cell_y..=n_max_cell_y {
                for &idx in &self.cells[iy as usize * FRAME_GRID_COLS + ix as usize] {
                    if check_levels {
                        let octave = keypoint_octave(scales.get(idx).copied().unwrap_or(1.0)) as isize;
                        if octave < min_level {
                            continue;
                        }
                        if max_level >= 0 && octave > max_level {
                            continue;
                        }
                    }

                    let kp = keypoints_xy[idx];
                    let dx = kp[0] - x;
                    let dy = kp[1] - y;
                    if dx.abs() < radius && dy.abs() < radius {
                        result.push(idx);
                    }
                }
            }
        }

        result
    }

    #[allow(dead_code)]
    pub(super) fn bounds(&self) -> (f32, f32, f32, f32) {
        (self.min_x, self.max_x, self.min_y, self.max_y)
    }
}

fn pos_in_grid(
    x: f32,
    y: f32,
    min_x: f32,
    min_y: f32,
    grid_w_inv: f32,
    grid_h_inv: f32,
) -> Option<(usize, usize)> {
    let col = ((x - min_x) * grid_w_inv).round() as isize;
    let row = ((y - min_y) * grid_h_inv).round() as isize;
    if !(0..FRAME_GRID_COLS as isize).contains(&col) || !(0..FRAME_GRID_ROWS as isize).contains(&row)
    {
        return None;
    }
    Some((col as usize, row as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn octave_from_scale(scale: f32) -> usize {
        match scale.round() as i32 {
            1 => 0,
            2 => 1,
            _ => 0,
        }
    }

    #[test]
    fn test_empty_grid() {
        let grid = KeypointGrid::new(&[], (0.0, 640.0, 0.0, 480.0));
        let result =
            grid.query_features_in_area(320.0, 240.0, 50.0, -1, -1, &[], &[], octave_from_scale);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_point_found() {
        let kps = [[100.0, 100.0]];
        let scales = [1.0];
        let grid = KeypointGrid::new(&kps, (0.0, 640.0, 0.0, 480.0));

        let result = grid.query_features_in_area(
            100.0,
            100.0,
            10.0,
            -1,
            -1,
            &kps,
            &scales,
            octave_from_scale,
        );
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_square_window_matches_orb_slam3() {
        let kps = [[65.0, 65.0], [75.0, 75.0], [85.0, 85.0]];
        let scales = [1.0, 1.0, 1.0];
        let grid = KeypointGrid::new(&kps, (0.0, 640.0, 0.0, 480.0));
        let mut result = grid.query_features_in_area(
            70.0,
            70.0,
            10.0,
            -1,
            -1,
            &kps,
            &scales,
            octave_from_scale,
        );
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_level_filter_matches_orb_slam3_bounds() {
        let kps = [[100.0, 100.0], [102.0, 100.0]];
        let scales = [1.0, 2.0];
        let grid = KeypointGrid::new(&kps, (0.0, 640.0, 0.0, 480.0));

        let result = grid.query_features_in_area(
            101.0,
            100.0,
            5.0,
            0,
            0,
            &kps,
            &scales,
            octave_from_scale,
        );
        assert_eq!(result, vec![0]);
    }
}
