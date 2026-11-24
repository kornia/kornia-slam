//! F3: rectifying a pair through `StereoRemap` allocates nothing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use kornia_3d::stereo::{CameraCalib, StereoRectifier};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;
use kornia_slam::rectify::{RemapQuantisation, StereoRemap};

/// Counts allocations on every thread while `COUNTING` is set.
struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

fn note_allocation() {
    if COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every call forwards to `System` with the caller's arguments; the
// counter has no effect on the memory returned.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        // SAFETY: forwarded unchanged; `layout` meets `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        // SAFETY: as `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_allocation();
        // SAFETY: `ptr` came from this allocator, which is `System`'s.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator, which is `System`'s.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn calib(k1: f64) -> CameraCalib {
    CameraCalib {
        width: 752,
        height: 480,
        fx: 458.0,
        fy: 457.0,
        cx: 370.0,
        cy: 250.0,
        distortion: PolynomialDistortion {
            k1,
            k2: 0.07,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 0.0,
            p2: 0.0,
        },
    }
}

#[test]
fn rectifying_a_pair_allocates_nothing() {
    let rectifier = StereoRectifier::from_calib(
        &calib(-0.28),
        &calib(-0.27),
        Mat3F64::IDENTITY,
        Vec3F64::new(-0.11, 0.0, 0.0),
    )
    .unwrap();
    let remap = StereoRemap::new(&rectifier, RemapQuantisation::Q10Trunc).unwrap();
    let size = ImageSize {
        width: 752,
        height: 480,
    };
    let raw = Image::new(size, (0..752 * 480).map(|i| (i % 251) as u8).collect()).unwrap();
    let (mut left, mut right) = (
        Image::from_size_val(size, 0u8).unwrap(),
        Image::from_size_val(size, 0u8).unwrap(),
    );
    remap.rectify(&raw, &raw, &mut left, &mut right).unwrap();
    COUNTING.store(true, Ordering::SeqCst);
    for _ in 0..3 {
        remap.rectify(&raw, &raw, &mut left, &mut right).unwrap();
    }
    COUNTING.store(false, Ordering::SeqCst);
    assert_eq!(ALLOCATIONS.load(Ordering::SeqCst), 0);
}
