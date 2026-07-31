use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use kornia_slam::map::Map;

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
                let mut map_guard = map.lock().unwrap();
                if job.imu_initialized {
                    map_guard.run_local_inertial_ba(&camera, job.imu_t_bc, job.gravity_world);
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
                } else {
                    map_guard.run_local_ba(&camera);
                }
                map_guard.cull();
            } // map_guard drops here, at the end of each loop iteration
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
