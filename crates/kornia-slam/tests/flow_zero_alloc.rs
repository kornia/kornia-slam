//! Card E, K8: a `RigFlow` step allocates nothing once warm, and its
//! survivors do not depend on the flow thread count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::{Image, ImageSize};
use kornia_slam::system::{AnchorGuard, ForwardBackward, GyroGuess, KltRetry, TrackRefill};
use kornia_slam::tracking::TrackingTimings;
use kornia_slam::tracking::optical_flow::{
    FlowStepInput, FlowSurvivor, RigFlow, RigFlowCamera, RigFlowSettings, TrackSet,
};

/// Counts allocations on every thread while `COUNTING` is set.
struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
/// The allocator is process-wide: one test at a time.
static SERIAL: Mutex<()> = Mutex::new(());

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

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const CAMERAS: usize = 2;
const WARM_UP: usize = 10;
const COUNTED: usize = 50;

/// Bright squares on a dark background, shifted by `shift` pixels; camera 1
/// sees them offset by half a period.
fn image(camera: usize, shift: [i64; 2]) -> Image<u8, 1> {
    const PERIOD: i64 = 60;
    const SIDE: i64 = 20;
    let offset = camera as i64 * PERIOD / 2;
    let data = (0..WIDTH * HEIGHT)
        .map(|i| {
            let x = (i % WIDTH) as i64 - shift[0] - offset;
            let y = (i / WIDTH) as i64 - shift[1];
            let inside = x.rem_euclid(PERIOD) < SIDE && y.rem_euclid(PERIOD) < SIDE;
            if inside { 220 } else { 30 }
        })
        .collect();
    Image::new(
        ImageSize {
            width: WIDTH,
            height: HEIGHT,
        },
        data,
    )
    .expect("image size matches its data")
}

/// A slow back-and-forth drift, one pixel per frame.
fn shift(frame: usize) -> [i64; 2] {
    let phase = (frame % 16) as i64;
    let s = if phase < 8 { phase } else { 16 - phase };
    [s, s / 2]
}

fn settings(threads: usize) -> RigFlowSettings {
    RigFlowSettings {
        threads,
        first_core: None,
        window: 11,
        predicted: (0, 8),
        unpredicted: (0, 8),
        epsilon: 0.01,
        retry: KltRetry::Off,
        forward_backward: ForwardBackward::Enabled {
            max_round_trip_px: 0.6,
            max_iter: 4,
            every_n_frames: 1,
        },
        refill: TrackRefill::Off,
        anchor_guard: AnchorGuard::Off,
        gyro_guess: GyroGuess::Off,
    }
}

fn cameras() -> Vec<RigFlowCamera> {
    let camera = PinholeCamera {
        fx: 400.0,
        fy: 400.0,
        cx: 320.0,
        cy: 240.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };
    (0..CAMERAS)
        .map(|c| RigFlowCamera {
            camera: camera.clone(),
            camera_from_primary: (c > 0).then_some(Pose3d::IDENTITY),
        })
        .collect()
}

/// Square corners of every camera, spawned on the layout's shards.
fn tracks(threads: usize) -> TrackSet {
    let mut tracks = TrackSet::with_flow_threads(threads, CAMERAS);
    for camera in 0..CAMERAS {
        let offset = camera as f32 * 30.0;
        for row in 1..7 {
            for col in 1..9 {
                let pixel = [col as f32 * 60.0 + offset, row as f32 * 60.0];
                tracks.spawn_on(pixel, camera);
            }
        }
    }
    tracks
}

/// Replays `frames` frames on `threads` flow threads, counting allocations
/// inside each step after the warm-up; `after_step` sees every survivor list.
fn replay(threads: usize, frames: usize, mut after_step: impl FnMut(usize, &[FlowSurvivor])) {
    let mut flow = RigFlow::new(settings(threads), cameras()).expect("flow workers start");
    let mut tracks = tracks(threads);
    let mut timings = TrackingTimings::default();
    for frame in 0..frames {
        let images: Vec<Image<u8, 1>> = (0..CAMERAS).map(|c| image(c, shift(frame))).collect();
        let refs: Vec<&Image<u8, 1>> = images.iter().collect();
        flow.prepare(&refs, frame as u64, &mut timings)
            .expect("frames prepare");
        let input = FlowStepInput {
            frame_idx: frame as u64,
            pose_world_to_primary: Some(Pose3d::IDENTITY),
            primary_rotation: None,
        };
        let counting = frame >= WARM_UP;
        COUNTING.store(counting, Ordering::SeqCst);
        let survivors_len = flow
            .step(&tracks, &input, &mut timings)
            .expect("flow step")
            .survivors
            .len();
        COUNTING.store(false, Ordering::SeqCst);
        let survivors = flow.output().survivors.clone();
        assert_eq!(survivors.len(), survivors_len);
        after_step(frame, &survivors);
        if tracks.advance(survivors).is_err() {
            panic!("flow survivors must be tracks of the set");
        }
    }
}

#[test]
fn a_warm_flow_step_does_not_allocate() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for threads in 1..=3 {
        ALLOCATIONS.store(0, Ordering::SeqCst);
        replay(threads, WARM_UP + COUNTED, |_, _| {});
        assert_eq!(
            ALLOCATIONS.load(Ordering::SeqCst),
            0,
            "allocations in {COUNTED} warm steps with {threads} flow threads"
        );
    }
}

#[test]
fn survivors_do_not_depend_on_the_thread_count() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let run = |threads| {
        let mut all = Vec::new();
        replay(threads, 30, |_, survivors| all.push(survivors.to_vec()));
        all
    };
    let one = run(1);
    assert!(one.iter().any(|frame| !frame.is_empty()), "nothing tracked");
    assert_eq!(run(2), one);
    assert_eq!(run(3), one);
}
