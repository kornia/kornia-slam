pub mod euroc;
#[allow(dead_code)]
pub mod hilti;
pub mod stereo_calib;

// The Bouguet stereo rectifier lives upstream in kornia-3d; re-export it here so
// dataset code can refer to `crate::datasets::StereoRectifier`. The EuRoC `T_BS`
// adapter that feeds it lives with the EuRoC reader in `euroc`.
pub use euroc::{EurocDataset, rectifier_from_euroc};
pub use kornia_3d::stereo::StereoRectifier;
pub use stereo_calib::StereoCalib;
