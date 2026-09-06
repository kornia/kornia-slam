//! Spatial grid for fast keypoint radius queries.

/// A 2D grid that bins keypoints by their image position for O(1) spatial queries.
pub(super) struct KeypointGrid {
    cells: Vec<Vec<usize>>,
    cell_w: f32,
    cell_h: f32,
    n_cols: usize,
    n_rows: usize,
    img_w: f32,
    img_h: f32,
}

impl KeypointGrid {
    /// Builds a grid over `keypoints_xy` (each `[x, y]`) for an image of size `img_w x img_h`.
    ///
    /// Each cell is `cell_size x cell_size` pixels. Points outside the image are clamped.
    pub(super) fn new(keypoints_xy: &[[f32; 2]], img_w: f32, img_h: f32, cell_size: f32) -> Self {
        let n_cols = (img_w / cell_size).ceil() as usize;
        let n_rows = (img_h / cell_size).ceil() as usize;
        let n_cells = n_cols * n_rows;
        let mut cells = vec![Vec::new(); n_cells];

        for (i, kp) in keypoints_xy.iter().enumerate() {
            let col = ((kp[0] / cell_size) as usize).min(n_cols - 1);
            let row = ((kp[1] / cell_size) as usize).min(n_rows - 1);
            cells[row * n_cols + col].push(i);
        }

        Self {
            cells,
            cell_w: cell_size,
            cell_h: cell_size,
            n_cols,
            n_rows,
            img_w,
            img_h,
        }
    }

    /// Returns indices of keypoints within `radius` pixels of `(x, y)`.
    pub(super) fn query_radius(
        &self,
        x: f32,
        y: f32,
        radius: f32,
        keypoints_xy: &[[f32; 2]],
    ) -> Vec<usize> {
        let r_sq = radius * radius;

        let col_min = ((x - radius).max(0.0) / self.cell_w) as usize;
        let col_max =
            (((x + radius).min(self.img_w - 1.0) / self.cell_w) as usize).min(self.n_cols - 1);
        let row_min = ((y - radius).max(0.0) / self.cell_h) as usize;
        let row_max =
            (((y + radius).min(self.img_h - 1.0) / self.cell_h) as usize).min(self.n_rows - 1);

        let mut result = Vec::new();
        for r in row_min..=row_max {
            for c in col_min..=col_max {
                for &idx in &self.cells[r * self.n_cols + c] {
                    let kp = keypoints_xy[idx];
                    let dx = kp[0] - x;
                    let dy = kp[1] - y;
                    if dx * dx + dy * dy <= r_sq {
                        result.push(idx);
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_grid() {
        let grid = KeypointGrid::new(&[], 640.0, 480.0, 64.0);
        let result = grid.query_radius(320.0, 240.0, 50.0, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_point_found() {
        let kps = [[100.0, 100.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result = grid.query_radius(100.0, 100.0, 10.0, &kps);
        assert_eq!(result, vec![0]);

        let result = grid.query_radius(500.0, 400.0, 10.0, &kps);
        assert!(result.is_empty());
    }

    #[test]
    fn test_radius_boundary() {
        let kps = [[50.0, 50.0], [60.0, 50.0], [100.0, 50.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 32.0);

        let mut result = grid.query_radius(55.0, 50.0, 15.0, &kps);
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_corner_clamping() {
        let kps = [[0.0, 0.0], [639.0, 479.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result = grid.query_radius(0.0, 0.0, 5.0, &kps);
        assert_eq!(result, vec![0]);

        let result = grid.query_radius(639.0, 479.0, 5.0, &kps);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_multiple_points_per_cell() {
        let kps = [[10.0, 10.0], [12.0, 11.0], [15.0, 13.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let mut result = grid.query_radius(12.0, 12.0, 20.0, &kps);
        result.sort();
        assert_eq!(result, vec![0, 1, 2]);
    }
}
