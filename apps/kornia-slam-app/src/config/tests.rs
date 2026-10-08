use std::path::Path;

use kornia_slam::system::PipelineDefinition;
use kornia_slam::{CameraSelection, ConfigError};

use super::*;

fn parse(text: &str) -> Result<RunConfig, Invalid> {
    RunConfig::from_ron_str(text)
}

fn vocabulary(run: &RunConfig) -> Option<&Path> {
    let PipelineDefinition::OrbSlam(orb) = &run.system.pipeline;
    orb.loop_closing.vocabulary()
}

#[test]
fn reads_the_proposed_syntax() {
    let run = parse(
        r#"(
            source: Euroc((data: "../data/MH_01_easy", max_frames: 500)),
            system: (version: 1, sensors: (cameras: Stereo, imu: true)),
        )"#,
    )
    .unwrap();
    let SourceConfig::Euroc(euroc) = &run.source else {
        panic!("expected EuRoC, got {:?}", run.source);
    };
    assert_eq!(euroc.data, Path::new("../data/MH_01_easy"));
    assert_eq!((euroc.start_frame, euroc.max_frames), (0, 500));
    assert_eq!(run.system.sensors.cameras, CameraSelection::Stereo);
    assert!(run.system.sensors.imu);
}

#[test]
fn an_omitted_system_runs_the_default_pipeline() {
    let run = parse(r#"(source: Euroc((data: "d")))"#).unwrap();
    assert_eq!(
        run.system.to_ron_string().unwrap(),
        PipelineConfig::default().to_ron_string().unwrap()
    );
}

#[test]
fn source_settings_keep_their_defaults() {
    let run = parse(r#"(source: Mcap((path: "r.mcap")))"#).unwrap();
    let SourceConfig::Mcap(mcap) = run.source else {
        panic!()
    };
    assert_eq!(
        (mcap.channel.as_str(), mcap.right_channel.as_str()),
        ("mono_left", "mono_right")
    );
    assert!(mcap.calib.is_none());

    let run = parse(r#"(source: Hilti((data: "d", calib: "c.yaml")))"#).unwrap();
    let SourceConfig::Hilti(hilti) = run.source else {
        panic!()
    };
    assert!(hilti.rotate_180);

    let run = parse("(source: Oakd(()))").unwrap();
    let SourceConfig::Oakd(oakd) = run.source else {
        panic!()
    };
    assert_eq!((oakd.width, oakd.height, oakd.fps), (640, 400, 30.0));

    let run = parse("(source: Uvc((fx: 600, fy: 600, cx: 320, cy: 240)))").unwrap();
    let SourceConfig::Uvc(uvc) = run.source else {
        panic!()
    };
    assert_eq!((uvc.index, uvc.width, uvc.height), (0, 640, 480));
    assert_eq!((uvc.k1, uvc.k2, uvc.p1, uvc.p2), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn required_settings_must_be_given() {
    for text in [
        "(system: (version: 1))",
        "(source: Euroc(()))",
        r#"(source: Hilti((data: "d")))"#,
        "(source: Uvc((fx: 600, fy: 600, cx: 320)))",
    ] {
        assert!(matches!(parse(text), Err(Invalid::Parse(_))), "{text}");
    }
}

#[test]
fn unknown_fields_are_rejected() {
    for text in [
        r#"(source: Euroc((data: "d")), output: "x")"#,
        r#"(source: Euroc((data: "d", stereo: true)))"#,
        r#"(source: Kitti((data: "d")))"#,
        r#"(source: Euroc((data: "d")), system: (version: 1, sensors: (lidar: true)))"#,
    ] {
        assert!(matches!(parse(text), Err(Invalid::Parse(_))), "{text}");
    }
}

#[test]
fn another_system_version_is_reported_before_its_fields() {
    let err =
        parse(r#"(source: Euroc((data: "d")), system: (version: 2, future: (x: 1)))"#).unwrap_err();
    assert!(matches!(
        err,
        Invalid::System(ConfigError::UnsupportedVersion { found: 2, .. })
    ));
}

#[test]
fn system_settings_are_validated() {
    let err = parse(
        r#"(source: Euroc((data: "d")), system: (version: 1,
            pipeline: OrbSlam((loop_closing: Enabled(vocabulary: "v.txt")))))"#,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Invalid::System(ConfigError::LoopClosingWithoutMetricScale)
    ));
}

#[test]
fn stereo_recordings_and_cameras_need_calibration() {
    let stereo = "system: (version: 1, sensors: (cameras: Stereo))";
    for source in [r#"Mcap((path: "r.mcap"))"#, "Oakd(())"] {
        let text = format!("(source: {source}, {stereo})");
        assert!(matches!(parse(&text), Err(Invalid::Source(_))), "{text}");
    }
    assert!(
        parse(&format!(
            r#"(source: Mcap((path: "r.mcap", calib: "c.yaml")), {stereo})"#
        ))
        .is_ok()
    );
}

#[test]
fn sensors_the_source_cannot_provide_are_rejected_before_opening() {
    for (source, sensors, message) in [
        (
            r#"Hilti((data: "d", calib: "c"))"#,
            "(cameras: Stereo)",
            "monocular images only",
        ),
        (
            "Uvc((fx: 600, fy: 600, cx: 320, cy: 240))",
            "(cameras: Stereo)",
            "monocular images only",
        ),
        (
            r#"Hilti((data: "d", calib: "c"))"#,
            "(imu: true)",
            "no IMU data",
        ),
        (r#"Mcap((path: "r.mcap"))"#, "(imu: true)", "no IMU data"),
        ("Oakd(())", "(imu: true)", "no IMU data"),
        (
            "Uvc((fx: 600, fy: 600, cx: 320, cy: 240))",
            "(imu: true)",
            "no IMU data",
        ),
    ] {
        let text = format!("(source: {source}, system: (version: 1, sensors: {sensors}))");
        match parse(&text) {
            Err(Invalid::Source(error)) => assert!(error.contains(message), "{text}: {error}"),
            other => panic!("{text}: expected a source error, got {:?}", other.err()),
        }
    }
    let euroc = r#"(source: Euroc((data: "d")), system: (version: 1, sensors: (cameras: Stereo, imu: true)))"#;
    assert!(parse(euroc).is_ok());
}

#[test]
fn camera_settings_are_range_checked() {
    for text in [
        "(source: Oakd((fps: 0)))",
        "(source: Oakd((width: 0)))",
        "(source: Uvc((fx: -1, fy: 600, cx: 320, cy: 240)))",
        "(source: Uvc((fx: 600, fy: 600, cx: 320, cy: 240, height: 0)))",
    ] {
        assert!(matches!(parse(text), Err(Invalid::Source(_))), "{text}");
    }
}

/// A run file in a temporary directory, removed on drop.
struct TempRun {
    dir: PathBuf,
}

impl TempRun {
    fn new(name: &str, text: &str) -> (Self, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("kornia-slam-run-{}-{name}", std::process::id()));
        let nested = dir.join("configs");
        std::fs::create_dir_all(&nested).unwrap();
        let path = nested.join("run.ron");
        std::fs::write(&path, text).unwrap();
        (Self { dir }, path)
    }
}

impl Drop for TempRun {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn relative_paths_resolve_against_the_run_file() {
    let (temp, path) = TempRun::new(
        "relative",
        r#"(source: Hilti((data: "../data/floor_2", calib: "/calib/chain.yaml")))"#,
    );
    let run = RunConfig::from_ron_file(&path).unwrap();
    let SourceConfig::Hilti(hilti) = &run.source else {
        panic!()
    };
    assert_eq!(hilti.data, temp.dir.join("configs/../data/floor_2"));
    assert_eq!(hilti.calib, Path::new("/calib/chain.yaml"));
}

#[test]
fn the_vocabulary_resolves_against_the_run_file() {
    let (temp, path) = TempRun::new(
        "vocabulary",
        r#"(source: Euroc((data: "/data/MH_01_easy")),
            system: (version: 1, sensors: (cameras: Stereo, imu: true),
                pipeline: OrbSlam((loop_closing: Enabled(vocabulary: "../weights/ORBvoc.txt")))))"#,
    );
    let run = RunConfig::from_ron_file(&path).unwrap();
    assert_eq!(
        vocabulary(&run),
        Some(temp.dir.join("configs/../weights/ORBvoc.txt").as_path())
    );
    let SourceConfig::Euroc(euroc) = &run.source else {
        panic!()
    };
    assert_eq!(euroc.data, Path::new("/data/MH_01_easy"));
}

#[test]
fn optional_paths_resolve_too() {
    let (temp, path) = TempRun::new(
        "optional",
        r#"(source: Mcap((path: "r.mcap", calib: "c.yaml")),
            system: (version: 1, sensors: (cameras: Stereo)))"#,
    );
    let run = RunConfig::from_ron_file(&path).unwrap();
    let SourceConfig::Mcap(mcap) = &run.source else {
        panic!()
    };
    let configs = temp.dir.join("configs");
    assert_eq!(mcap.path, configs.join("r.mcap"));
    assert_eq!(
        mcap.calib.as_deref(),
        Some(configs.join("c.yaml").as_path())
    );
}

/// A bare file name has an empty parent; its paths stay as written, relative
/// to the working directory, which is the file's directory.
#[test]
fn a_bare_file_name_keeps_paths_as_written() {
    let mut run = parse(r#"(source: Euroc((data: "MH_01_easy")))"#).unwrap();
    run.resolve_paths(Path::new("run.ron").parent().unwrap());
    let SourceConfig::Euroc(euroc) = &run.source else {
        panic!()
    };
    assert_eq!(euroc.data, Path::new("MH_01_easy"));
}

#[test]
fn errors_name_the_run_file() {
    let err = RunConfig::from_ron_file("/nonexistent/run.ron").unwrap_err();
    assert!(
        err.to_string().starts_with("/nonexistent/run.ron: "),
        "{err}"
    );

    let (_temp, path) = TempRun::new("invalid", "(source: Oakd((fps: 0)))");
    let err = RunConfig::from_ron_file(&path).unwrap_err();
    assert!(matches!(err, RunConfigError::Source { .. }));
    assert!(err.to_string().contains("run.ron: source: fps"), "{err}");
}
