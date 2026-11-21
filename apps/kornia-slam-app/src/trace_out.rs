//! `--trace-out`: per-frame stage timings and, inside the traced frame range,
//! the tracking inputs a frontend benchmark replays.
//!
//! Files: `rig.csv` (cameras as tracking sees them), `timings.csv` (every
//! frame), `frames.csv` and `tracks.csv` (traced frames), `local_map.bin`
//! (traced frames, little-endian 60-byte records: `u32 frame_idx, u32
//! map_point_idx, f32 x, y, z, f32 min_distance, max_distance, [u8; 32]
//! descriptor`) and `images/<frame_idx>_<camera>.u8` (traced frames, the
//! row-major gray images tracking received, after any rectification).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_image::Image;
use kornia_slam::{FrameTrace, FrontendObservation, SensorRig, TrackingStatus, TrackingTimings};

pub(crate) struct TraceWriter {
    dir: std::path::PathBuf,
    rig_written: bool,
    images_dir: std::path::PathBuf,
    timings: BufWriter<File>,
    frames: BufWriter<File>,
    tracks: BufWriter<File>,
    local_map: BufWriter<File>,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn pose_fields(pose: &Pose3d) -> String {
    let r = pose.rotation.to_cols_array();
    let t = pose.translation;
    let mut out = String::new();
    // Row-major rotation, then translation.
    for row in 0..3 {
        for col in 0..3 {
            out.push_str(&format!("{:.12},", r[col * 3 + row]));
        }
    }
    out.push_str(&format!("{:.12},{:.12},{:.12}", t.x, t.y, t.z));
    out
}

fn camera_fields(camera: &PinholeCamera) -> String {
    format!(
        "{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
        camera.fx, camera.fy, camera.cx, camera.cy, camera.k1, camera.k2, camera.p1, camera.p2
    )
}

/// `camera,width,height,fx,fy,cx,cy,k1,k2,p1,p2,stereo_baseline_m` and the
/// primary-from-camera pose; camera 0 is the primary. A rectified stereo
/// pair has one row and its baseline; a rig one row per camera.
fn write_rig(dir: &Path, rig: &SensorRig, primary_size: [usize; 2]) -> std::io::Result<()> {
    let mut out = BufWriter::new(File::create(dir.join("rig.csv"))?);
    writeln!(
        out,
        "camera,width,height,fx,fy,cx,cy,k1,k2,p1,p2,stereo_baseline_m,\
         r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz"
    )?;
    let baseline = rig.stereo_baseline_m.unwrap_or(0.0);
    if rig.cameras.is_empty() {
        writeln!(
            out,
            "0,{},{},{},{baseline:.12},{}",
            primary_size[0],
            primary_size[1],
            camera_fields(&rig.camera),
            pose_fields(&Pose3d::IDENTITY)
        )?;
    }
    for (index, camera) in rig.cameras.iter().enumerate() {
        writeln!(
            out,
            "{index},{},{},{},{baseline:.12},{}",
            camera.size.width,
            camera.size.height,
            camera_fields(&camera.camera),
            pose_fields(&camera.primary_from_camera)
        )?;
    }
    out.flush()
}

impl TraceWriter {
    pub(crate) fn create(dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let images_dir = dir.join("images");
        std::fs::create_dir_all(&images_dir)?;
        let open = |name: &str| File::create(dir.join(name)).map(BufWriter::new);
        let mut timings = open("timings.csv")?;
        writeln!(
            timings,
            "frame_idx,status,orb_ms,overlap_ms,frontend_ms,klt_build_ms,klt_track_ms,\
             projection_search_ms,consistency_filter_ms,pnp_ms,local_refine_ms,\
             keyframe_insert_ms,tracking_ms"
        )?;
        let mut frames = open("frames.csv")?;
        writeln!(
            frames,
            "frame_idx,timestamp_sec,trusted_prediction,tracked,keyframe,n_local,n_tracks,\
             cand_r00,cand_r01,cand_r02,cand_r10,cand_r11,cand_r12,cand_r20,cand_r21,cand_r22,\
             cand_tx,cand_ty,cand_tz,\
             pose_r00,pose_r01,pose_r02,pose_r10,pose_r11,pose_r12,pose_r20,pose_r21,pose_r22,\
             pose_tx,pose_ty,pose_tz"
        )?;
        let mut tracks = open("tracks.csv")?;
        writeln!(tracks, "frame_idx,track_id,x,y,map_point_idx")?;
        Ok(Self {
            dir: dir.to_path_buf(),
            rig_written: false,
            images_dir,
            timings,
            frames,
            tracks,
            local_map: open("local_map.bin")?,
        })
    }

    pub(crate) fn write_timings(
        &mut self,
        frame_idx: usize,
        status: TrackingStatus,
        frontend: &FrontendObservation,
        timings: &TrackingTimings,
        tracking: Duration,
    ) -> std::io::Result<()> {
        let orb: Vec<String> = frontend
            .orb_per_camera
            .iter()
            .map(|d| format!("{:.3}", ms(*d)))
            .collect();
        writeln!(
            self.timings,
            "{frame_idx},{status:?},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            orb.join(";"),
            ms(frontend.overlap),
            ms(frontend.duration),
            ms(timings.klt_build),
            ms(timings.klt_track),
            ms(timings.projection_search),
            ms(timings.consistency_filter),
            ms(timings.pnp),
            ms(timings.local_refine),
            ms(timings.keyframe_insert),
            ms(tracking),
        )
    }

    /// Writes `trace` under the source's frame index `frame_idx`.
    pub(crate) fn write_trace(
        &mut self,
        frame_idx: usize,
        trace: &FrameTrace,
    ) -> std::io::Result<()> {
        writeln!(
            self.frames,
            "{},{:.9},{},{},{},{},{},{},{}",
            frame_idx,
            trace.timestamp_sec,
            u8::from(trace.trusted_prediction),
            u8::from(trace.tracked),
            u8::from(trace.keyframe),
            trace.local_map.len(),
            trace.tracks.len(),
            pose_fields(&trace.candidate_pose),
            pose_fields(&trace.pose),
        )?;
        for track in &trace.tracks {
            let mp = track.map_point_idx.map_or(-1, |idx| idx as i64);
            writeln!(
                self.tracks,
                "{},{},{:.6},{:.6},{mp}",
                frame_idx, track.track_id, track.pixel[0], track.pixel[1]
            )?;
        }
        let frame_idx = frame_idx as u32;
        for point in &trace.local_map {
            let mut record = [0u8; 60];
            record[0..4].copy_from_slice(&frame_idx.to_le_bytes());
            record[4..8].copy_from_slice(&(point.map_point_idx as u32).to_le_bytes());
            let floats = [
                point.position[0] as f32,
                point.position[1] as f32,
                point.position[2] as f32,
                point.min_distance as f32,
                point.max_distance as f32,
            ];
            for (i, value) in floats.iter().enumerate() {
                record[8 + 4 * i..12 + 4 * i].copy_from_slice(&value.to_le_bytes());
            }
            record[28..60].copy_from_slice(&point.descriptor);
            self.local_map.write_all(&record)?;
        }
        Ok(())
    }

    /// Writes the images of traced frame `frame_idx`, primary first, and
    /// `rig.csv` with the first of them.
    pub(crate) fn write_images(
        &mut self,
        rig: &SensorRig,
        frame_idx: usize,
        images: &[&Image<u8, 1>],
    ) -> std::io::Result<()> {
        if !self.rig_written
            && let Some(primary) = images.first()
        {
            write_rig(&self.dir, rig, [primary.width(), primary.height()])?;
            self.rig_written = true;
        }
        for (camera, image) in images.iter().enumerate() {
            std::fs::write(
                self.images_dir.join(format!("{frame_idx}_{camera}.u8")),
                image.as_slice(),
            )?;
        }
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> std::io::Result<()> {
        self.timings.flush()?;
        self.frames.flush()?;
        self.tracks.flush()?;
        self.local_map.flush()
    }
}
