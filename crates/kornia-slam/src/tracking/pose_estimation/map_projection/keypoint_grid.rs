//! Spatial grid for fast keypoint radius queries.

use std::slice;

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

    /// Yields indices of keypoints within `radius` pixels of `(x, y)`, cell by cell in
    /// row-major order and ascending within a cell.
    pub(super) fn query_radius<'a>(
        &'a self,
        x: f32,
        y: f32,
        radius: f32,
        keypoints_xy: &'a [[f32; 2]],
    ) -> impl Iterator<Item = usize> {
        let col_min = ((x - radius).max(0.0) / self.cell_w) as usize;
        let col_max =
            (((x + radius).min(self.img_w - 1.0) / self.cell_w) as usize).min(self.n_cols - 1);
        let row_min = ((y - radius).max(0.0) / self.cell_h) as usize;
        let row_max =
            (((y + radius).min(self.img_h - 1.0) / self.cell_h) as usize).min(self.n_rows - 1);

        RadiusQuery {
            grid: self,
            keypoints_xy,
            x,
            y,
            r_sq: radius * radius,
            col_min,
            col_max,
            row_max,
            next_row: row_min,
            next_col: col_min,
            cell: [].iter(),
        }
    }
}

struct RadiusQuery<'a> {
    grid: &'a KeypointGrid,
    keypoints_xy: &'a [[f32; 2]],
    x: f32,
    y: f32,
    r_sq: f32,
    col_min: usize,
    col_max: usize,
    row_max: usize,
    next_row: usize,
    next_col: usize,
    cell: slice::Iter<'a, usize>,
}

impl Iterator for RadiusQuery<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            for &idx in self.cell.by_ref() {
                let kp = self.keypoints_xy[idx];
                let dx = kp[0] - self.x;
                let dy = kp[1] - self.y;
                if dx * dx + dy * dy <= self.r_sq {
                    return Some(idx);
                }
            }
            if self.next_row > self.row_max || self.next_col > self.col_max {
                return None;
            }
            self.cell = self.grid.cells[self.next_row * self.grid.n_cols + self.next_col].iter();
            if self.next_col < self.col_max {
                self.next_col += 1;
            } else {
                self.next_col = self.col_min;
                self.next_row += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The allocating scan `query_radius` replaced, kept to pin its output and order.
    fn query_radius_reference(
        grid: &KeypointGrid,
        x: f32,
        y: f32,
        radius: f32,
        keypoints_xy: &[[f32; 2]],
    ) -> Vec<usize> {
        let r_sq = radius * radius;

        let col_min = ((x - radius).max(0.0) / grid.cell_w) as usize;
        let col_max =
            (((x + radius).min(grid.img_w - 1.0) / grid.cell_w) as usize).min(grid.n_cols - 1);
        let row_min = ((y - radius).max(0.0) / grid.cell_h) as usize;
        let row_max =
            (((y + radius).min(grid.img_h - 1.0) / grid.cell_h) as usize).min(grid.n_rows - 1);

        let mut result = Vec::new();
        for r in row_min..=row_max {
            for c in col_min..=col_max {
                for &idx in &grid.cells[r * grid.n_cols + c] {
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

    /// Deterministic uniform samples in [lo, hi).
    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((self.0 >> 40) as f32 / (1u32 << 24) as f32)
        }
    }

    #[test]
    fn test_empty_grid() {
        let grid = KeypointGrid::new(&[], 640.0, 480.0, 64.0);
        let result: Vec<usize> = grid.query_radius(320.0, 240.0, 50.0, &[]).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_point_found() {
        let kps = [[100.0, 100.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result: Vec<usize> = grid.query_radius(100.0, 100.0, 10.0, &kps).collect();
        assert_eq!(result, vec![0]);

        let result: Vec<usize> = grid.query_radius(500.0, 400.0, 10.0, &kps).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn test_radius_boundary() {
        let kps = [[50.0, 50.0], [60.0, 50.0], [100.0, 50.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 32.0);

        let mut result: Vec<usize> = grid.query_radius(55.0, 50.0, 15.0, &kps).collect();
        result.sort();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_corner_clamping() {
        let kps = [[0.0, 0.0], [639.0, 479.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result: Vec<usize> = grid.query_radius(0.0, 0.0, 5.0, &kps).collect();
        assert_eq!(result, vec![0]);

        let result: Vec<usize> = grid.query_radius(639.0, 479.0, 5.0, &kps).collect();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_multiple_points_per_cell() {
        let kps = [[10.0, 10.0], [12.0, 11.0], [15.0, 13.0]];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let mut result: Vec<usize> = grid.query_radius(12.0, 12.0, 20.0, &kps).collect();
        result.sort();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn query_radius_matches_the_allocating_scan() {
        const RADII: [f32; 6] = [0.0, 0.5, 15.0, 30.0, 60.0, 1000.0];
        const MARGIN: f32 = 80.0;

        let mut rng = Lcg(177);
        for (img_w, img_h, cell_size) in [(752.0, 480.0, 64.0), (640.0, 480.0, 37.5)] {
            let edges = |extent: f32| {
                (0..=(extent / cell_size).ceil() as usize).map(move |k| k as f32 * cell_size)
            };

            let mut sample = || {
                [
                    rng.uniform(-MARGIN, img_w + MARGIN),
                    rng.uniform(-MARGIN, img_h + MARGIN),
                ]
            };

            let mut kps: Vec<[f32; 2]> = (0..300).map(|_| sample()).collect();
            kps.extend(edges(img_w).flat_map(|x| edges(img_h).map(move |y| [x, y])));
            kps.extend([
                [img_w - 1.0, img_h - 1.0],
                [img_w, img_h],
                [f32::NAN, 100.0],
                [100.0, f32::NAN],
                [f32::NAN, f32::NAN],
            ]);
            let grid = KeypointGrid::new(&kps, img_w, img_h, cell_size);

            let queries: Vec<[f32; 2]> = (0..300)
                .map(|_| sample())
                .chain(kps.iter().copied())
                .collect();
            for [x, y] in queries {
                for radius in RADII {
                    assert_eq!(
                        grid.query_radius(x, y, radius, &kps).collect::<Vec<_>>(),
                        query_radius_reference(&grid, x, y, radius, &kps),
                        "({x}, {y}) radius {radius} on {img_w}x{img_h} with {cell_size} px cells"
                    );
                }
            }
        }
    }

    #[test]
    fn nan_keypoints_are_never_returned() {
        let kps = [
            [f32::NAN, 10.0],
            [10.0, f32::NAN],
            [f32::NAN, f32::NAN],
            [10.0, 10.0],
        ];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);
        assert_eq!(grid.cells[0], [0, 1, 2, 3]);

        for radius in [0.0, 15.0, 1000.0] {
            let result: Vec<usize> = grid.query_radius(10.0, 10.0, radius, &kps).collect();
            assert_eq!(result, vec![3], "radius {radius}");
        }
    }

    #[test]
    fn results_follow_cell_order_then_index() {
        let kps = [
            [80.0, 80.0],
            [50.0, 50.0],
            [80.0, 50.0],
            [50.0, 80.0],
            [55.0, 55.0],
            [94.0, 64.0], // exactly `radius` away
            [34.0, 63.0], // just outside
        ];
        let grid = KeypointGrid::new(&kps, 640.0, 480.0, 64.0);

        let result: Vec<usize> = grid.query_radius(64.0, 64.0, 30.0, &kps).collect();
        assert_eq!(result, vec![1, 4, 2, 3, 0, 5]);
    }
}
