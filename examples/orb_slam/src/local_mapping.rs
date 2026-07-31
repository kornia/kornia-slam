use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use kornia_3d::ba::BaParams;
use kornia_3d::ba_schur::bundle_adjust_schur;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_slam::map::Map;
use kornia_slam::vi_ba_schur::{ViBaParams, visual_inertial_bundle_adjust};

pub struct KeyframeJob {
    pub kf_idx: usize,
    pub imu_initialized: bool,
    pub imu_t_bc: Option<Pose3d>,
    pub gravity_world: Vec3F64,
}
pub struct LocalMappingHandle {
    sender: Option<mpsc::Sender<KeyframeJob>>,
    join_handle: Option<JoinHandle<()>>,
}
impl LocalMappingHandle {
    pub fn spawn(map: Arc<Mutex<Map>>, camera: PinholeCamera) -> Self {
        let (sender, receiver) = mpsc::channel::<KeyframeJob>();

        let join_handle = thread::spawn(move || {
            for job in receiver {
                if job.imu_initialized {
                    // Snapshot phase: brief lock, just copying out the
                    // window's poses/velocities/biases/observations.
                    let snapshot = {
                        let mut map_guard = map.lock().unwrap();
                        map_guard.snapshot_local_inertial_ba(&camera)
                    }; // map_guard dropped here — unlocked for the solve below

                    let Some(snapshot) = snapshot else {
                        continue;
                    };

                    // Solve phase: no lock held. This is the expensive part
                    // (up to 50 LM iterations) — running it outside the lock
                    // is what actually lets tracking's frequent self.map.lock()
                    // calls proceed uncontended while this runs.
                    let result = visual_inertial_bundle_adjust(
                        &snapshot.vi_keyframes,
                        &snapshot.points,
                        &snapshot.observations,
                        &snapshot.imu_edges,
                        &camera,
                        &ViBaParams {
                            imu_t_bc: job.imu_t_bc,
                            gravity: job.gravity_world,
                            max_iterations: 50,
                            ..ViBaParams::default()
                        },
                    );

                    // Write-back phase: brief lock again.
                    if let Ok(result) = result {
                        let mut map_guard = map.lock().unwrap();
                        if map_guard.apply_local_inertial_ba_result(&snapshot, &result) {
                            if let Some(kf) = map_guard.get_keyframe(job.kf_idx) {
                                println!(
                                    "kf accel bias: {:3}, {:3}, {:3}",
                                    kf.imu_bias.accel.x, kf.imu_bias.accel.y, kf.imu_bias.accel.z
                                );
                                println!(
                                    "kf gyro bias: {:3}, {:3}, {:3}",
                                    kf.imu_bias.gyro.x, kf.imu_bias.gyro.y, kf.imu_bias.gyro.z
                                );
                            }
                            map_guard.cull();
                        } else {
                            println!(
                                "[local_mapping] discarded stale VI-BA result for kf={} (map transformed while solving)",
                                job.kf_idx
                            );
                        }
                    }
                } else {
                    let snapshot = {
                        let map_guard = map.lock().unwrap();
                        map_guard.snapshot_local_ba(&camera)
                    };

                    let Some(snapshot) = snapshot else {
                        continue;
                    };

                    let result = bundle_adjust_schur(
                        &snapshot.poses,
                        &snapshot.points,
                        &snapshot.observations,
                        &camera,
                        &BaParams::default(),
                    );

                    if let Ok(result) = result {
                        let mut map_guard = map.lock().unwrap();
                        if map_guard.apply_local_ba_result(&snapshot, &result) {
                            map_guard.cull();
                        } else {
                            println!(
                                "[local_mapping] discarded stale BA result for kf={} (map transformed while solving)",
                                job.kf_idx
                            );
                        }
                    }
                }
            }
        });

        Self {
            sender: Some(sender),
            join_handle: Some(join_handle),
        }
    }

    pub fn submit(&self, job: KeyframeJob) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(job);
        }
    }
}

impl Drop for LocalMappingHandle {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}
