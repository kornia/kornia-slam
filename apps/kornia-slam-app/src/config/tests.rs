use std::path::Path;

use kornia_slam::system::PipelineDefinition;
use kornia_slam::{CameraSelection, ConfigError, SystemConfig};

use super::*;

fn parse(text: &str) -> Result<RunConfig, Invalid> {
    RunConfig::from_ron_str(text)
}

fn parse_error(text: &str) -> ron::Error {
    match parse(text) {
        Err(Invalid::Parse(error)) => error.code,
        other => panic!("{text}: expected a parse error, got {:?}", other.err()),
    }
}

fn source_error(text: &str) -> String {
    match parse(text) {
        Err(Invalid::Source(message)) => message,
        other => panic!("{text}: expected a source error, got {:?}", other.err()),
    }
}

fn vocabulary(run: &RunConfig) -> Option<&Path> {
    let PipelineDefinition::OrbSlam(orb) = &run.system.pipeline;
    orb.loop_closing.vocabulary()
}

fn configs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs")
}

#[test]
fn reads_a_run_file() {
    let run = parse(
        r#"(
            source: Euroc((data: "../data/euroc/MH_01_easy", max_frames: 500)),
            system: (version: 1, sensors: (cameras: Stereo, imu: true)),
        )"#,
    )
    .unwrap();
    let SourceConfig::Euroc(euroc) = &run.source else {
        panic!("expected EuRoC, got {:?}", run.source);
    };
    assert_eq!(euroc.data, Path::new("../data/euroc/MH_01_easy"));
    assert_eq!((euroc.start_frame, euroc.max_frames), (0, 500));
    assert_eq!(run.system.sensors.cameras, CameraSelection::Stereo);
    assert!(run.system.sensors.imu);
}

#[test]
fn an_omitted_system_runs_the_default_pipeline() {
    let run = parse(r#"(source: Euroc((data: "d")))"#).unwrap();
    assert_eq!(
        run.system.to_ron_string().unwrap(),
        SystemConfig::default().to_ron_string().unwrap()
    );
}

#[test]
fn source_settings_keep_their_defaults() {
    let run = parse(r#"(source: Mcap((path: "r.mcap")))"#).unwrap();
    let SourceConfig::Mcap(mcap) = run.source else {
        panic!("expected MCAP, got {:?}", run.source);
    };
    assert_eq!(
        (mcap.channel.as_str(), mcap.right_channel.as_str()),
        ("mono_left", "mono_right")
    );
    assert!(mcap.calib.is_none());

    let run = parse(r#"(source: Hilti((data: "d", calib: "c.yaml")))"#).unwrap();
    let SourceConfig::Hilti(hilti) = run.source else {
        panic!("expected Hilti, got {:?}", run.source);
    };
    assert!(hilti.rotate_180);

    let run = parse("(source: Oakd(()))").unwrap();
    let SourceConfig::Oakd(oakd) = run.source else {
        panic!("expected OAK-D, got {:?}", run.source);
    };
    assert_eq!((oakd.width, oakd.height, oakd.fps), (640, 400, 30.0));

    let run = parse("(source: Uvc((fx: 600, fy: 600, cx: 320, cy: 240)))").unwrap();
    let SourceConfig::Uvc(uvc) = run.source else {
        panic!("expected UVC, got {:?}", run.source);
    };
    assert_eq!((uvc.index, uvc.width, uvc.height), (0, 640, 480));
    assert_eq!((uvc.k1, uvc.k2, uvc.p1, uvc.p2), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn missing_and_unknown_settings_are_rejected() {
    for (text, field) in [
        ("(system: (version: 1))", "source"),
        ("(source: Euroc(()))", "data"),
        (r#"(source: Hilti((data: "d")))"#, "calib"),
        ("(source: Uvc((fx: 600, fy: 600, cx: 320)))", "cy"),
    ] {
        assert!(
            matches!(parse_error(text), ron::Error::MissingStructField { field: f, .. } if f == field),
            "{text}"
        );
    }
    for (text, field) in [
        (r#"(source: Euroc((data: "d")), output: "x")"#, "output"),
        (r#"(source: Euroc((data: "d", stereo: true)))"#, "stereo"),
    ] {
        assert!(
            matches!(parse_error(text), ron::Error::NoSuchStructField { found, .. } if found == field),
            "{text}"
        );
    }
    assert!(matches!(
        parse_error(r#"(source: Kitti((data: "d")))"#),
        ron::Error::NoSuchEnumVariant { found, .. } if found == "Kitti"
    ));
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
fn sources_are_checked_before_opening() {
    let run = |source: &str, sensors: &str| {
        format!("(source: {source}, system: (version: 1, sensors: {sensors}))")
    };
    let mono = "()";
    let stereo = "(cameras: Stereo)";
    let imu = "(imu: true)";
    let hilti = r#"Hilti((data: "d", calib: "c"))"#;
    let mcap = r#"Mcap((path: "r.mcap"))"#;
    let uvc = "Uvc((fx: 600, fy: 600, cx: 320, cy: 240))";
    for (source, sensors, message) in [
        (mcap, stereo, "needs `calib`"),
        ("Oakd(())", stereo, "needs `calib`"),
        (hilti, stereo, "monocular images only"),
        (uvc, stereo, "monocular images only"),
        (hilti, imu, "needs `rotate_180: false`"),
        (r#"TumVi((data: "d"))"#, stereo, "monocular images only"),
        (mcap, imu, "no IMU data"),
        ("Oakd(())", imu, "no IMU data"),
        (uvc, imu, "no IMU data"),
        ("Oakd((fps: 0))", mono, "fps is 0"),
        ("Oakd((width: 0))", mono, "width is 0"),
        ("Uvc((fx: -1, fy: 600, cx: 320, cy: 240))", mono, "fx is -1"),
        (
            "Uvc((fx: 600, fy: 600, cx: 320, cy: 240, height: 0))",
            mono,
            "height is 0",
        ),
    ] {
        let text = run(source, sensors);
        let error = source_error(&text);
        assert!(error.contains(message), "{text}: {error}");
    }
    for (source, sensors) in [
        (r#"Mcap((path: "r.mcap", calib: "c.yaml"))"#, stereo),
        (r#"TumVi((data: "d"))"#, imu),
        (r#"Hilti((data: "d", calib: "c", rotate_180: false))"#, imu),
        (r#"Euroc((data: "d"))"#, "(cameras: Stereo, imu: true)"),
    ] {
        assert!(parse(&run(source, sensors)).is_ok(), "{source} {sensors}");
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
        "paths",
        r#"(source: Mcap((path: "../data/r.mcap", calib: "c.yaml")),
            system: (version: 1, sensors: (cameras: Stereo),
                pipeline: OrbSlam((loop_closing: Enabled(vocabulary: "/weights/ORBvoc.txt")))))"#,
    );
    let run = RunConfig::from_ron_file(&path).unwrap();
    let SourceConfig::Mcap(mcap) = &run.source else {
        panic!("expected MCAP, got {:?}", run.source);
    };
    let configs = temp.dir.join("configs");
    assert_eq!(mcap.path, configs.join("../data/r.mcap"));
    assert_eq!(
        mcap.calib.as_deref(),
        Some(configs.join("c.yaml").as_path())
    );
    assert_eq!(vocabulary(&run), Some(Path::new("/weights/ORBvoc.txt")));
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

#[test]
fn shipped_run_files_load() {
    let mut loaded = 0;
    for entry in std::fs::read_dir(configs_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "ron") {
            RunConfig::from_ron_file(&path)
                .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
            loaded += 1;
        }
    }
    assert!(loaded > 0, "no run files in {}", configs_dir().display());
}

/// `euroc.ron` documents every system default, so it must stay equal to them.
#[test]
fn euroc_run_file_lists_the_system_defaults() {
    let run = RunConfig::from_ron_file(configs_dir().join("euroc.ron")).unwrap();
    assert_eq!(
        run.system.to_ron_string().unwrap(),
        SystemConfig::default().to_ron_string().unwrap(),
        "configs/euroc.ron no longer lists the system defaults"
    );
}
