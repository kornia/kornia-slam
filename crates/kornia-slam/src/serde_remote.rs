//! Serde representations of kornia-rs configuration types used in pipeline files.
//!
//! Each module mirrors one upstream type for `#[serde(with = ...)]`. Omitted
//! fields take the default of the field that uses the module, which can differ
//! from the upstream type's own default, so a module serves one context only.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Mirrors `$remote` field by field. Construction is exhaustive, so a field
/// added upstream fails to compile here instead of being silently dropped.
macro_rules! remote_settings {
    ($module:ident, $remote:ty, $default:expr, { $($field:ident: $ty:ty),* $(,)? }) => {
        pub(crate) mod $module {
            use super::*;

            #[derive(Serialize, Deserialize)]
            #[serde(default, deny_unknown_fields)]
            struct Settings {
                $($field: $ty),*
            }

            impl Default for Settings {
                fn default() -> Self {
                    Self::from(&$default)
                }
            }

            impl From<&$remote> for Settings {
                fn from(value: &$remote) -> Self {
                    Self { $($field: value.$field),* }
                }
            }

            impl From<Settings> for $remote {
                fn from(settings: Settings) -> Self {
                    Self { $($field: settings.$field),* }
                }
            }

            pub(crate) fn serialize<S: Serializer>(
                value: &$remote,
                serializer: S,
            ) -> Result<S::Ok, S::Error> {
                Settings::from(value).serialize(serializer)
            }

            pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
                deserializer: D,
            ) -> Result<$remote, D::Error> {
                Settings::deserialize(deserializer).map(Into::into)
            }
        }
    };
}

remote_settings!(
    orb_match,
    kornia_imgproc::features::OrbMatchConfig,
    kornia_imgproc::features::OrbMatchConfig::default(),
    {
        nn_ratio: f32,
        th_low: u32,
        check_orientation: bool,
        histo_length: usize,
    }
);

remote_settings!(
    two_view_triangulation,
    kornia_3d::pose::TriangulationConfig,
    crate::initialization::two_view::TwoViewInitConfig::default().triangulation_config,
    {
        min_parallax_deg: f64,
        max_midpoint_gap: f64,
        max_reprojection_error: f64,
        min_cheirality_count: usize,
        cheirality_ambiguity_max: f64,
    }
);

remote_settings!(
    local_projection,
    crate::tracking::pose_estimation::map_projection::ProjectionMatchConfig,
    crate::tracking::pose_estimation::map_projection::MapProjectionConfig::default()
        .local_projection,
    {
        min_depth: f64,
        search_radius: f32,
        max_hamming: u32,
    }
);

pub(crate) mod robust_kernel {
    use kornia_3d::ransac::RobustKernelKind;

    use super::*;

    #[derive(Serialize, Deserialize)]
    enum Kernel {
        Identity,
        Huber,
        Cauchy,
        Tukey,
    }

    pub(crate) fn serialize<S: Serializer>(
        value: &RobustKernelKind,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            RobustKernelKind::Identity => Kernel::Identity,
            RobustKernelKind::Huber => Kernel::Huber,
            RobustKernelKind::Cauchy => Kernel::Cauchy,
            RobustKernelKind::Tukey => Kernel::Tukey,
        }
        .serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<RobustKernelKind, D::Error> {
        Ok(match Kernel::deserialize(deserializer)? {
            Kernel::Identity => RobustKernelKind::Identity,
            Kernel::Huber => RobustKernelKind::Huber,
            Kernel::Cauchy => RobustKernelKind::Cauchy,
            Kernel::Tukey => RobustKernelKind::Tukey,
        })
    }
}

/// PnP RANSAC settings of loop verification.
pub(crate) mod loop_pnp_ransac {
    use kornia_3d::pnp::RansacParams;
    use kornia_3d::ransac::SPRTConfig;

    use super::*;

    #[derive(Serialize, Deserialize)]
    #[serde(default, deny_unknown_fields)]
    struct Settings {
        max_iterations: usize,
        reproj_threshold_px: f32,
        confidence: f32,
        random_seed: Option<u64>,
        refine: bool,
        sprt: Option<Sprt>,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    #[allow(non_snake_case)]
    struct Sprt {
        epsilon: f64,
        delta: f64,
        t_M: f64,
        t_m: f64,
    }

    impl Default for Settings {
        fn default() -> Self {
            Self::from(&crate::loop_closure::LoopVerificationConfig::default().pnp_ransac)
        }
    }

    impl From<&RansacParams> for Settings {
        fn from(value: &RansacParams) -> Self {
            Self {
                max_iterations: value.max_iterations,
                reproj_threshold_px: value.reproj_threshold_px,
                confidence: value.confidence,
                random_seed: value.random_seed,
                refine: value.refine,
                sprt: value.sprt.map(|sprt| Sprt {
                    epsilon: sprt.epsilon,
                    delta: sprt.delta,
                    t_M: sprt.t_M,
                    t_m: sprt.t_m,
                }),
            }
        }
    }

    impl From<Settings> for RansacParams {
        fn from(settings: Settings) -> Self {
            Self {
                max_iterations: settings.max_iterations,
                reproj_threshold_px: settings.reproj_threshold_px,
                confidence: settings.confidence,
                random_seed: settings.random_seed,
                refine: settings.refine,
                sprt: settings.sprt.map(|sprt| SPRTConfig {
                    epsilon: sprt.epsilon,
                    delta: sprt.delta,
                    t_M: sprt.t_M,
                    t_m: sprt.t_m,
                }),
            }
        }
    }

    pub(crate) fn serialize<S: Serializer>(
        value: &RansacParams,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Settings::from(value).serialize(serializer)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<RansacParams, D::Error> {
        Settings::deserialize(deserializer).map(Into::into)
    }
}
