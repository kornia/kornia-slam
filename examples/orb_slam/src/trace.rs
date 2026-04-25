use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use crate::pipeline::PipelineTrace;
use kornia_slam::estimation::map_projection::map_projection_reject_reason_name;
use kornia_slam::estimation::two_view::two_view_reject_reason_name;
use kornia_slam::system::{SystemMode, TrackingStatus};

pub struct TraceWriter {
    frames: BufWriter<File>,
    init: BufWriter<File>,
    tracking: BufWriter<File>,
    map: BufWriter<File>,
}

impl TraceWriter {
    pub fn new(prefix: impl AsRef<Path>) -> io::Result<Self> {
        let prefix = prefix.as_ref();
        let mut writer = Self {
            frames: BufWriter::new(File::create(suffixed_path(prefix, "frames"))?),
            init: BufWriter::new(File::create(suffixed_path(prefix, "init"))?),
            tracking: BufWriter::new(File::create(suffixed_path(prefix, "tracking"))?),
            map: BufWriter::new(File::create(suffixed_path(prefix, "map"))?),
        };
        writer.write_headers()?;
        Ok(writer)
    }

    pub fn write_frame(
        &mut self,
        frame_idx: usize,
        timestamp_ns: i64,
        feature_count: usize,
        map_points: usize,
        active_map_points: usize,
        keyframes: usize,
        status: TrackingStatus,
        elapsed_ms: f64,
        trace: &PipelineTrace,
    ) -> io::Result<()> {
        writeln!(
            self.frames,
            "{frame_idx},{timestamp_ns},{},{},{},{feature_count},{map_points},{keyframes},{},{elapsed_ms:.6}",
            mode_name(trace.mode_before),
            mode_name(trace.mode_after),
            status_name(status),
            trace.tracked_inliers,
        )?;

        if let Some(init) = &trace.init {
            let reject_reason = init
                .report
                .reject_reason
                .map(two_view_reject_reason_name)
                .unwrap_or("none");
            let r = &init.report;
            let mut ng = [0usize; 8];
            for (i, v) in r.candidate_ngood.iter().take(8).enumerate() {
                ng[i] = *v;
            }
            let best_cand_idx: i32 = r.best_candidate_idx.map(|i| i as i32).unwrap_or(-1);
            let ran_i = if r.two_view_ran { 1 } else { 0 };
            let acc_i = if r.two_view_accepted { 1 } else { 0 };
            writeln!(
                self.init,
                "{fir},{fic},{fcr},{fcc},{mtc},{mdl},{inl},{tri},{mdp:.12},{rej},\
{ran},{att},{acc},{tm},{sh:.4},{sf:.4},{rh:.4},{ih},{ifi},\
{n0},{n1},{n2},{n3},{n4},{n5},{n6},{n7},\
{bg},{sbg},{bci},{bpd:.4},{ns}",
                fir = init.frame_idx_ref,
                fic = init.frame_idx_cur,
                fcr = init.feature_count_ref,
                fcc = init.feature_count_cur,
                mtc = r.matches,
                mdl = r.model,
                inl = r.inliers,
                tri = r.triangulated,
                mdp = r.median_depth.unwrap_or(0.0),
                rej = reject_reason,
                ran = ran_i,
                att = ran_i,
                acc = acc_i,
                tm = r.model,
                sh = r.orb_slam3_sh,
                sf = r.orb_slam3_sf,
                rh = r.orb_slam3_rh,
                ih = r.inliers_h,
                ifi = r.inliers_f,
                n0 = ng[0],
                n1 = ng[1],
                n2 = ng[2],
                n3 = ng[3],
                n4 = ng[4],
                n5 = ng[5],
                n6 = ng[6],
                n7 = ng[7],
                bg = r.triangulated,
                sbg = r.second_best_good,
                bci = best_cand_idx,
                bpd = r.median_parallax_deg,
                ns = 0,
            )?;
        }

        if let Some(tracking) = &trace.tracking {
            let reject_reason = tracking
                .reject_reason
                .map(map_projection_reject_reason_name)
                .unwrap_or("none");
            writeln!(
                self.tracking,
                "{frame_idx},{},{},{},{},{},{},{}",
                tracking.projection_matches,
                tracking.projection_pnp_inliers,
                tracking.reference_matches,
                tracking.reference_correspondences,
                tracking.local_projection_matches,
                tracking.local_pnp_inliers,
                reject_reason,
            )?;
        }

        writeln!(
            self.map,
            "{frame_idx},{keyframes},{map_points},{active_map_points}"
        )?;
        Ok(())
    }

    fn write_headers(&mut self) -> io::Result<()> {
        writeln!(
            self.frames,
            "frame_idx,timestamp_ns,mode_before,mode_after,status,feature_count,map_points,keyframes,tracked_inliers,elapsed_ms"
        )?;
        writeln!(
            self.init,
            "frame_idx_ref,frame_idx_cur,feature_count_ref,feature_count_cur,descriptor_matches,model,inliers,triangulated,median_depth,reject_reason,\
tv_ran,tv_attempted,tv_accepted,tv_model,sh,sf,rh,inliers_h,inliers_f,\
ngood_0,ngood_1,ngood_2,ngood_3,ngood_4,ngood_5,ngood_6,ngood_7,\
best_good,second_best_good,best_cand_idx,best_parallax_deg,n_similar"
        )?;
        writeln!(
            self.tracking,
            "frame_idx,projection_matches,projection_pnp_inliers,reference_matches,reference_correspondences,local_projection_matches,local_pnp_inliers,reject_reason"
        )?;
        writeln!(self.map, "frame_idx,keyframes,map_points,active_map_points")?;
        Ok(())
    }
}

fn suffixed_path(prefix: &Path, name: &str) -> PathBuf {
    PathBuf::from(format!("{}_{}.csv", prefix.display(), name))
}

fn mode_name(mode: SystemMode) -> &'static str {
    match mode {
        SystemMode::Bootstrap => "bootstrap",
        SystemMode::Tracking => "tracking",
    }
}

fn status_name(status: TrackingStatus) -> &'static str {
    match status {
        TrackingStatus::Tracked => "tracked",
        TrackingStatus::Skipped => "skipped",
        TrackingStatus::KeyframeAccepted => "keyframe_accepted",
    }
}
