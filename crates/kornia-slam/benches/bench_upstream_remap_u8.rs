//! `upstream::kornia_imgproc::remap_u8` against kornia-rs `remap_u8` and,
//! with `opencv_bench`, OpenCV's `remap` on `convertMaps` `CV_16SC2` maps.
//! One EuRoC `cam0` rectification: 752×480, radtan (k1 −0.283). The tables
//! and OpenCV's fixed maps are built once, outside the timed loop.
//!
//! `RAYON_NUM_THREADS=1 taskset -c 8 cargo bench -j2 -p kornia-slam
//! --features bench_internals,opencv_bench --bench bench_upstream_remap_u8`

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use kornia_3d::stereo::{CameraCalib, StereoRectifier};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;
use kornia_imgproc::interpolation::{InterpolationMode, remap_u8};
use kornia_slam::upstream::kornia_imgproc::remap_u8::{
    RemapQuantisation, RemapTable, remap_u8_table,
};

const SIZE: ImageSize = ImageSize {
    width: 752,
    height: 480,
};

/// EuRoC `cam0` / `cam1` (`mav0/cam*/sensor.yaml`).
fn rectifier() -> StereoRectifier {
    let calib = |fx, fy, cx, cy, k1, k2, p1, p2| CameraCalib {
        width: SIZE.width,
        height: SIZE.height,
        fx,
        fy,
        cx,
        cy,
        distortion: PolynomialDistortion {
            k1,
            k2,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1,
            p2,
        },
    };
    let left = calib(
        458.654,
        457.296,
        367.215,
        248.375,
        -0.28340811,
        0.07395907,
        0.00019359,
        1.76187114e-05,
    );
    let right = calib(
        457.587,
        456.134,
        379.999,
        255.238,
        -0.28368365,
        0.07451284,
        -0.00010473,
        -3.55590700e-05,
    );
    // Row-major rotation of `T_BS`.
    let t_bs = |r: [f64; 9], t: [f64; 3]| {
        let col = |c: usize| Vec3F64::new(r[c], r[3 + c], r[6 + c]);
        (
            Mat3F64::from_cols(col(0), col(1), col(2)),
            Vec3F64::new(t[0], t[1], t[2]),
        )
    };
    let (r_l, t_l) = t_bs(
        [
            0.0148655429818,
            -0.999880929698,
            0.00414029679422,
            0.999557249008,
            0.0149672133247,
            0.025715529948,
            -0.0257744366974,
            0.00375618835797,
            0.999660727178,
        ],
        [-0.0216401454975, -0.064676986768, 0.00981073058949],
    );
    let (r_r, t_r) = t_bs(
        [
            0.0125552670891,
            -0.999755099723,
            0.0182237714554,
            0.999598781151,
            0.0130119051815,
            0.0251588363115,
            -0.0253898008918,
            0.0179005838253,
            0.999517347078,
        ],
        [-0.0198435579556, 0.0453689425024, 0.00786212447038],
    );
    let r_rt = r_r.transpose();
    StereoRectifier::from_calib(&left, &right, r_rt * r_l, r_rt * (t_l - t_r)).unwrap()
}

/// A textured frame: smooth shading plus fine detail.
fn frame() -> Image<u8, 1> {
    let data = (0..SIZE.width * SIZE.height)
        .map(|i| {
            let (x, y) = ((i % SIZE.width) as f32, (i / SIZE.width) as f32);
            (128.0 + 60.0 * (x * 0.05).sin() * (y * 0.07).cos() + 40.0 * (x * 0.9 + y * 1.3).sin())
                as u8
        })
        .collect();
    Image::new(SIZE, data).unwrap()
}

fn bench_remap(c: &mut Criterion) {
    let rectifier = rectifier();
    let (map_x, map_y) = rectifier.left_maps();
    let src = frame();
    let mut dst = Image::from_size_val(SIZE, 0u8).unwrap();
    let mut group = c.benchmark_group("remap_u8_752x480");
    group.bench_function("kornia_rs_remap_u8", |b| {
        b.iter(|| {
            remap_u8(
                black_box(&src),
                &mut dst,
                map_x,
                map_y,
                InterpolationMode::Bilinear,
            )
            .unwrap();
            black_box(&dst);
        });
    });
    for (name, quant) in [
        ("table_q10", RemapQuantisation::Q10Trunc),
        ("table_q5", RemapQuantisation::OpenCvQ5),
    ] {
        let table = RemapTable::from_maps(map_x, map_y, SIZE, quant).unwrap();
        group.bench_function(name, |b| {
            b.iter(|| {
                remap_u8_table(black_box(&src), &mut dst, &table).unwrap();
                black_box(&dst);
            });
        });
    }

    #[cfg(feature = "opencv_bench")]
    opencv_arm::bench(&mut group, &src, map_x, map_y);

    group.finish();
}

#[cfg(feature = "opencv_bench")]
mod opencv_arm {
    use super::*;
    use criterion::BenchmarkGroup;
    use criterion::measurement::WallTime;
    use opencv::{core, imgproc, prelude::*};

    fn mat<T: core::DataType + Copy>(img: &Image<T, 1>, typ: i32) -> core::Mat {
        let mut mat = core::Mat::zeros(img.height() as i32, img.width() as i32, typ)
            .unwrap()
            .to_mat()
            .unwrap();
        mat.data_typed_mut::<T>()
            .unwrap()
            .copy_from_slice(img.as_slice());
        mat
    }

    pub(super) fn bench(
        group: &mut BenchmarkGroup<'_, WallTime>,
        src: &Image<u8, 1>,
        map_x: &Image<f32, 1>,
        map_y: &Image<f32, 1>,
    ) {
        core::set_num_threads(1).unwrap();
        let cv_src = mat(src, core::CV_8UC1);
        let (mut xy, mut frac) = (core::Mat::default(), core::Mat::default());
        imgproc::convert_maps(
            &mat(map_x, core::CV_32FC1),
            &mat(map_y, core::CV_32FC1),
            &mut xy,
            &mut frac,
            core::CV_16SC2,
            false,
        )
        .unwrap();
        let mut out = core::Mat::default();
        group.bench_function("opencv_remap_16sc2", |b| {
            b.iter(|| {
                imgproc::remap(
                    black_box(&cv_src),
                    &mut out,
                    &xy,
                    &frac,
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    core::Scalar::all(0.0),
                )
                .unwrap();
                black_box(&out);
            });
        });
    }
}

criterion_group!(benches, bench_remap);
criterion_main!(benches);
