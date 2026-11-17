//! `--dump-map`: the live landmarks at shutdown, for offline map-quality
//! checks such as counting duplicate landmarks.

use std::io::Write;
use std::path::Path;

use kornia_slam::mapping::MapPoint;

/// Writes `landmark_idx,x,y,z,observations,descriptor` per live landmark, the
/// position in the world frame (metres) and the descriptor as hex. Slots are
/// append-only, so the index identifies a landmark in the final map.
pub fn write(path: &Path, map_points: &[MapPoint]) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "landmark_idx,x,y,z,observations,descriptor")?;
    for (idx, mp) in map_points.iter().enumerate().filter(|(_, mp)| !mp.culled) {
        let descriptor: String = mp.descriptor.iter().map(|b| format!("{b:02x}")).collect();
        writeln!(
            out,
            "{},{:.6},{:.6},{:.6},{},{descriptor}",
            idx,
            mp.position.x,
            mp.position.y,
            mp.position.z,
            mp.observations().len(),
        )?;
    }
    out.flush()
}
