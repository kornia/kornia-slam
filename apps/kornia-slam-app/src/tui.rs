//! Ratatui TUI for the ORB-SLAM example.
//!
//! Three panels:
//!   * top status bar (frame counters, status, FPS)
//!   * camera image (grayscale half-block render of the latest frame)
//!   * bird's-eye view of the trajectory + camera heading (X right, Z forward,
//!     Y-down gravity dropped)

use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::ExecutableCommand;
use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::widgets::{Block, Borders, Paragraph};

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

// ── Color palette ────────────────────────────────────────────────────────

const C_BORDER: Color = Color::Rgb(80, 120, 180);
const C_TITLE: Color = Color::Rgb(140, 220, 255);
const C_LABEL: Color = Color::Rgb(170, 170, 180);
const C_VALUE: Color = Color::Rgb(240, 240, 240);
const C_TRAJ: Color = Color::Rgb(80, 200, 255);
const C_CAMERA: Color = Color::Rgb(255, 215, 60);
const C_TRACKED: Color = Color::Rgb(120, 220, 100);
const C_KEYFRAME: Color = Color::Rgb(255, 110, 220);
const C_SKIPPED: Color = Color::Rgb(220, 180, 60);
const C_KEY: Color = Color::Rgb(255, 215, 60);
const C_SYS: Color = Color::Rgb(200, 160, 240);
const C_DEBUG: Color = Color::Rgb(170, 200, 170);

const DEBUG_LOG_CAPACITY: usize = 200;

/// Tracking status, mirrored from `kornia_slam::TrackingStatus` so the TUI
/// module doesn't import library types in its widget code.
#[derive(Clone, Copy)]
pub enum TuiStatus {
    Tracked,
    KeyframeAccepted,
    Skipped,
}

impl TuiStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Tracked => "Tracked",
            Self::KeyframeAccepted => "KeyframeAccepted",
            Self::Skipped => "Skipped",
        }
    }
    fn color(self) -> Color {
        match self {
            Self::Tracked => C_TRACKED,
            Self::KeyframeAccepted => C_KEYFRAME,
            Self::Skipped => C_SKIPPED,
        }
    }
}

// ── Stderr redirect ──────────────────────────────────────────────────────

/// RAII guard that redirects process stderr to a file via `dup2` while the
/// TUI is active, and restores the original on drop.
///
/// The SLAM pipeline emits diagnostic `eprintln!` lines that would otherwise
/// corrupt the alternate-screen canvas.
pub struct StderrGuard {
    saved_fd: libc::c_int,
}

impl StderrGuard {
    pub fn redirect_to(path: &Path) -> io::Result<Self> {
        use std::os::unix::io::AsRawFd;
        let saved_fd = unsafe { libc::dup(libc::STDERR_FILENO) };
        if saved_fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let file = match std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
        {
            Ok(f) => f,
            Err(e) => {
                unsafe { libc::close(saved_fd) };
                return Err(e);
            }
        };
        let ret = unsafe { libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) };
        if ret < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(saved_fd) };
            return Err(err);
        }
        Ok(Self { saved_fd })
    }
}

impl Drop for StderrGuard {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved_fd, libc::STDERR_FILENO);
            libc::close(self.saved_fd);
        }
    }
}

pub fn setup_terminal(stderr_log: &Path) -> io::Result<(Tui, StderrGuard)> {
    let guard = StderrGuard::redirect_to(stderr_log)?;
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    stdout.execute(cursor::Hide)?;
    let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    Ok((terminal, guard))
}

pub fn restore_terminal(terminal: &mut Tui) -> io::Result<()> {
    disable_raw_mode()?;
    let backend = terminal.backend_mut();
    backend.execute(LeaveAlternateScreen)?;
    backend.execute(cursor::Show)?;
    terminal.show_cursor()?;
    Ok(())
}

// ── System stats sampler ─────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
pub struct SysStats {
    pub rss_mb: f64,
    pub peak_rss_mb: f64,
    pub threads: u32,
    pub cpu_pct: f64,
}

/// Reads `/proc/self/status` + `getrusage` and caches the result so we don't
/// re-hit the kernel on every frame. CPU% is computed as the delta of
/// utime+stime since the last full sample, over wall-clock delta.
#[derive(Default)]
pub struct SysStatsSampler {
    last_cpu_us: i64,
    last_wall: Option<Instant>,
    cached: Option<(Instant, SysStats)>,
}

impl SysStatsSampler {
    pub fn sample(&mut self) -> SysStats {
        if let Some((t, s)) = self.cached
            && t.elapsed() < Duration::from_millis(250)
        {
            return s;
        }
        let (rss_kb, peak_rss_kb, threads) = read_proc_status();
        let now_us = getrusage_total_us();
        let now_wall = Instant::now();
        let cpu_pct = match (self.last_wall, self.last_cpu_us) {
            (Some(prev_wall), prev_cpu) if prev_cpu > 0 => {
                let wall_us = now_wall.duration_since(prev_wall).as_micros() as i64;
                if wall_us > 0 {
                    let cpu_us = now_us - prev_cpu;
                    (100.0 * cpu_us as f64 / wall_us as f64).max(0.0)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        self.last_cpu_us = now_us;
        self.last_wall = Some(now_wall);
        let stats = SysStats {
            rss_mb: rss_kb as f64 / 1024.0,
            peak_rss_mb: peak_rss_kb as f64 / 1024.0,
            threads,
            cpu_pct,
        };
        self.cached = Some((now_wall, stats));
        stats
    }
}

fn read_proc_status() -> (u64, u64, u32) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss = 0u64;
    let mut peak = 0u64;
    let mut threads = 0u32;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss = parse_kb_field(rest);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            peak = parse_kb_field(rest);
        } else if let Some(rest) = line.strip_prefix("Threads:") {
            threads = rest.trim().parse().unwrap_or(0);
        }
    }
    (rss, peak, threads)
}

fn parse_kb_field(rest: &str) -> u64 {
    rest.split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or(0)
}

fn getrusage_total_us() -> i64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            let u = ru.ru_utime.tv_sec as i64 * 1_000_000 + ru.ru_utime.tv_usec as i64;
            let s = ru.ru_stime.tv_sec as i64 * 1_000_000 + ru.ru_stime.tv_usec as i64;
            u + s
        } else {
            0
        }
    }
}

// ── App state ────────────────────────────────────────────────────────────

/// Live view state shared between the SLAM loop and the renderer.
pub struct TuiApp {
    /// BEV trajectory (world X, Z).
    trajectory: Vec<[f64; 2]>,
    /// Current camera position in BEV.
    current: Option<[f64; 2]>,
    /// Current camera forward direction in BEV (unit vector).
    forward: Option<[f64; 2]>,
    /// Full 3D camera-in-world position (most recent frame).
    pos_world: Option<Vec3F64>,
    /// Full 3D camera-in-world forward axis (most recent frame).
    fwd_world: Option<Vec3F64>,
    /// Previous-frame 3D position, used to compute per-frame displacement.
    prev_pos_world: Option<Vec3F64>,
    /// Magnitude of the last frame-to-frame displacement (metres / frame).
    last_step_m: f64,
    /// Cumulative path length (sum of per-frame displacements).
    path_length_m: f64,
    pub frame_idx: usize,
    pub n_frames: usize,
    pub status: TuiStatus,
    pub kf_idx: usize,
    pub n_active_mp: usize,
    pub frame_ms: f64,
    pub mean_ms: f64,
    sys: SysStatsSampler,
    /// When true, render the debug log panel.
    pub debug_enabled: bool,
    /// Rolling ring of recent debug messages from the SLAM pipeline.
    debug_lines: VecDeque<String>,
}

impl TuiApp {
    pub fn new(n_frames: usize) -> Self {
        Self {
            trajectory: Vec::with_capacity(n_frames),
            current: None,
            forward: None,
            pos_world: None,
            fwd_world: None,
            prev_pos_world: None,
            last_step_m: 0.0,
            path_length_m: 0.0,
            frame_idx: 0,
            n_frames,
            status: TuiStatus::Skipped,
            kf_idx: 0,
            n_active_mp: 0,
            frame_ms: 0.0,
            mean_ms: 0.0,
            sys: SysStatsSampler::default(),
            debug_enabled: false,
            debug_lines: VecDeque::with_capacity(DEBUG_LOG_CAPACITY),
        }
    }

    pub fn push_debug_line(&mut self, line: String) {
        if self.debug_lines.len() == DEBUG_LOG_CAPACITY {
            self.debug_lines.pop_front();
        }
        self.debug_lines.push_back(line);
    }

    /// Update from the latest world-to-camera pose.
    ///
    /// Camera-in-world is `pose.inverse()`; its translation is the camera
    /// position, and `R^T * [0,0,1]` is the camera forward axis in world
    /// coordinates. We drop Y (gravity) to get the BEV.
    pub fn update_pose(&mut self, pose_world_to_cam: &Pose3d) {
        let cam_to_world = pose_world_to_cam.inverse();
        let t = cam_to_world.translation;
        let f_world = cam_to_world.rotation * Vec3F64::new(0.0, 0.0, 1.0);

        // BEV projection (drop Y).
        let pos_bev = [t.x, t.z];
        self.trajectory.push(pos_bev);
        self.current = Some(pos_bev);
        let len_xz = (f_world.x * f_world.x + f_world.z * f_world.z).sqrt();
        if len_xz > 1e-9 {
            self.forward = Some([f_world.x / len_xz, f_world.z / len_xz]);
        }

        // Per-frame displacement + cumulative path length.
        if let Some(prev) = self.prev_pos_world {
            let dx = t.x - prev.x;
            let dy = t.y - prev.y;
            let dz = t.z - prev.z;
            self.last_step_m = (dx * dx + dy * dy + dz * dz).sqrt();
            self.path_length_m += self.last_step_m;
        }
        self.prev_pos_world = Some(t);
        self.pos_world = Some(t);
        self.fwd_world = Some(f_world);
    }

    pub fn draw(&mut self, terminal: &mut Tui) -> io::Result<()> {
        let sys = self.sys.sample();
        let debug_enabled = self.debug_enabled;
        terminal.draw(|frame| {
            let mut constraints = vec![
                Constraint::Length(3), // header
                Constraint::Min(0),    // BEV
                Constraint::Length(3), // camera
                Constraint::Length(3), // system
            ];
            if debug_enabled {
                constraints.push(Constraint::Length(10)); // debug log
            }
            constraints.push(Constraint::Length(1)); // hint
            let chunks = Layout::vertical(constraints).split(frame.area());

            frame.render_widget(self.header(), chunks[0]);
            self.render_bev(frame, chunks[1]);
            frame.render_widget(self.camera_panel(), chunks[2]);
            frame.render_widget(sys_panel(sys), chunks[3]);
            if debug_enabled {
                frame.render_widget(self.debug_panel(), chunks[4]);
                frame.render_widget(self.hint(), chunks[5]);
            } else {
                frame.render_widget(self.hint(), chunks[4]);
            }
        })?;
        Ok(())
    }

    fn debug_panel(&self) -> Paragraph<'_> {
        let style = Style::default().fg(C_DEBUG);
        let lines: Vec<Line> = self
            .debug_lines
            .iter()
            .rev()
            .take(8)
            .rev()
            .map(|s| Line::from(Span::styled(s.as_str(), style)))
            .collect();
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(C_BORDER))
                .title(Span::styled(
                    " debug ",
                    Style::default().fg(C_TITLE).add_modifier(Modifier::BOLD),
                )),
        )
    }

    fn header(&self) -> Paragraph<'_> {
        let instant_fps = if self.frame_ms > 0.0 {
            1000.0 / self.frame_ms
        } else {
            0.0
        };
        let mean_fps = if self.mean_ms > 0.0 {
            1000.0 / self.mean_ms
        } else {
            0.0
        };
        let bold_value = Style::default().fg(C_VALUE).add_modifier(Modifier::BOLD);
        let label = Style::default().fg(C_LABEL);
        let line = Line::from(vec![
            Span::styled(" frame ", label),
            Span::styled(
                format!("{:>4}/{:<4}", self.frame_idx, self.n_frames),
                bold_value,
            ),
            Span::styled("  status: ", label),
            Span::styled(
                format!("{:<16}", self.status.label()),
                Style::default()
                    .fg(self.status.color())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("kf=", label),
            Span::styled(format!("{:<5}", self.kf_idx), Style::default().fg(C_VALUE)),
            Span::styled(" mp=", label),
            Span::styled(
                format!("{:<5}", self.n_active_mp),
                Style::default().fg(C_VALUE),
            ),
            Span::styled("  ", label),
            Span::styled(format!("{:>5.1} ms", self.frame_ms), bold_value),
            Span::styled(format!(" ({:>5.1} fps)", instant_fps), label),
            Span::styled("  mean ", label),
            Span::styled(
                format!("{:>5.1} ms", self.mean_ms),
                Style::default().fg(C_VALUE),
            ),
            Span::styled(format!(" ({:>5.1} fps)", mean_fps), label),
        ]);
        Paragraph::new(line).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(C_BORDER))
                .title(Span::styled(
                    " orb_slam ",
                    Style::default().fg(C_TITLE).add_modifier(Modifier::BOLD),
                )),
        )
    }

    fn render_bev(&self, frame: &mut ratatui::Frame<'_>, area: Rect) {
        let inner_w = area.width.saturating_sub(2);
        let inner_h = area.height.saturating_sub(2);
        let (xmin, xmax, zmin, zmax) = self.viewport_bounds(inner_w, inner_h);
        let traj = self.trajectory.clone();
        let camera = self.camera_triangle(xmax - xmin, zmax - zmin);

        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(C_BORDER))
                    .title(Span::styled(
                        " BEV (X right / Z forward) ",
                        Style::default().fg(C_TITLE).add_modifier(Modifier::BOLD),
                    )),
            )
            .marker(Marker::Braille)
            .x_bounds([xmin, xmax])
            .y_bounds([zmin, zmax])
            .paint(move |ctx| {
                for w in traj.windows(2) {
                    ctx.draw(&CanvasLine {
                        x1: w[0][0],
                        y1: w[0][1],
                        x2: w[1][0],
                        y2: w[1][1],
                        color: C_TRAJ,
                    });
                }
                if let Some([apex, left, right]) = camera {
                    for (p, q) in [(apex, left), (apex, right), (left, right)] {
                        ctx.draw(&CanvasLine {
                            x1: p[0],
                            y1: p[1],
                            x2: q[0],
                            y2: q[1],
                            color: C_CAMERA,
                        });
                    }
                }
            });
        frame.render_widget(canvas, area);
    }

    fn camera_panel(&self) -> Paragraph<'_> {
        let lbl = Style::default().fg(C_LABEL);
        let val = Style::default().fg(C_CAMERA).add_modifier(Modifier::BOLD);
        let (pos_str, heading_str, elev_str) = match (self.pos_world, self.fwd_world) {
            (Some(p), Some(f)) => {
                // Heading: angle of the forward axis in the world X/Z plane,
                // measured from +Z (forward) toward +X (right). Range (-180°, 180°].
                let heading = f.x.atan2(f.z).to_degrees();
                // Elevation: angle of the forward axis above the horizontal
                // plane. Y is down, so "up" is -Y. Range [-90°, 90°].
                let horiz = (f.x * f.x + f.z * f.z).sqrt();
                let elevation = (-f.y).atan2(horiz).to_degrees();
                (
                    format!("[{:>6.2}, {:>6.2}, {:>6.2}] m", p.x, p.y, p.z),
                    format!("{:>+7.1}°", heading),
                    format!("{:>+5.1}°", elevation),
                )
            }
            _ => ("—".into(), "—".into(), "—".into()),
        };
        let line = Line::from(vec![
            Span::styled(" pos ", lbl),
            Span::styled(pos_str, val),
            Span::styled("   heading ", lbl),
            Span::styled(heading_str, val),
            Span::styled("   elev ", lbl),
            Span::styled(elev_str, val),
            Span::styled("   step ", lbl),
            Span::styled(format!("{:>5.3} m", self.last_step_m), val),
            Span::styled("   path ", lbl),
            Span::styled(format!("{:>6.2} m", self.path_length_m), val),
        ]);
        Paragraph::new(line).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(C_BORDER))
                .title(Span::styled(
                    " camera ",
                    Style::default().fg(C_TITLE).add_modifier(Modifier::BOLD),
                )),
        )
    }

    fn hint(&self) -> Paragraph<'static> {
        let key = Style::default().fg(C_KEY).add_modifier(Modifier::BOLD);
        let lbl = Style::default().fg(C_LABEL);
        Paragraph::new(Line::from(vec![
            Span::styled(" q", key),
            Span::styled(" quit  ", lbl),
            Span::styled("Esc", key),
            Span::styled(" quit  ", lbl),
            Span::styled("Ctrl-C", key),
            Span::styled(" quit  ", lbl),
            Span::styled("d", key),
            Span::styled(" toggle debug", lbl),
        ]))
    }

    /// Auto-fit bounds with padding + aspect correction.
    ///
    /// Braille gives 2×4 sub-pixels per char, and terminal cells are roughly
    /// 1:2 (W:H), so per-sub-pixel both axes are ≈square. Effective resolution
    /// of the BEV panel is therefore `(cells_w * 2) × (cells_h * 4)` square
    /// pixels — we match the world-span aspect to this so circles look round.
    fn viewport_bounds(&self, cells_w: u16, cells_h: u16) -> (f64, f64, f64, f64) {
        let (mut xmin, mut xmax, mut zmin, mut zmax) = if self.trajectory.is_empty() {
            (-1.0, 1.0, -1.0, 1.0)
        } else {
            let (mut a, mut b, mut c, mut d) = (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            );
            for p in &self.trajectory {
                a = a.min(p[0]);
                b = b.max(p[0]);
                c = c.min(p[1]);
                d = d.max(p[1]);
            }
            (a, b, c, d)
        };

        let pad_x = ((xmax - xmin) * 0.15).max(0.5);
        let pad_z = ((zmax - zmin) * 0.15).max(0.5);
        xmin -= pad_x;
        xmax += pad_x;
        zmin -= pad_z;
        zmax += pad_z;

        let pixel_w = (cells_w as f64) * 2.0;
        let pixel_h = (cells_h as f64) * 4.0;
        if pixel_w > 0.0 && pixel_h > 0.0 {
            let span_x = xmax - xmin;
            let span_z = zmax - zmin;
            let aspect_data = span_x / span_z;
            let aspect_pixels = pixel_w / pixel_h;
            if aspect_data > aspect_pixels {
                let new_span_z = span_x / aspect_pixels;
                let cz = 0.5 * (zmin + zmax);
                zmin = cz - 0.5 * new_span_z;
                zmax = cz + 0.5 * new_span_z;
            } else {
                let new_span_x = span_z * aspect_pixels;
                let cx = 0.5 * (xmin + xmax);
                xmin = cx - 0.5 * new_span_x;
                xmax = cx + 0.5 * new_span_x;
            }
        }

        (xmin, xmax, zmin, zmax)
    }

    /// Isoceles triangle with apex along the camera forward axis.
    fn camera_triangle(&self, span_x: f64, span_z: f64) -> Option<[[f64; 2]; 3]> {
        let pos = self.current?;
        let fwd = self.forward?;
        let l = span_x.max(span_z) * 0.03;
        let w = l * 0.5;
        let perp = [-fwd[1], fwd[0]];
        let apex = [pos[0] + fwd[0] * l, pos[1] + fwd[1] * l];
        let left = [pos[0] + perp[0] * w, pos[1] + perp[1] * w];
        let right = [pos[0] - perp[0] * w, pos[1] - perp[1] * w];
        Some([apex, left, right])
    }
}

fn sys_panel(s: SysStats) -> Paragraph<'static> {
    let label = Style::default().fg(C_LABEL);
    let value = Style::default().fg(C_SYS).add_modifier(Modifier::BOLD);
    let line = Line::from(vec![
        Span::styled(" ram ", label),
        Span::styled(format!("{:>6.1} MB", s.rss_mb), value),
        Span::styled("   peak ", label),
        Span::styled(format!("{:>6.1} MB", s.peak_rss_mb), value),
        Span::styled("   threads ", label),
        Span::styled(format!("{}", s.threads), value),
        Span::styled("   cpu ", label),
        Span::styled(format!("{:>5.1}%", s.cpu_pct), value),
    ]);
    Paragraph::new(line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(C_BORDER))
            .title(Span::styled(
                " system ",
                Style::default().fg(C_TITLE).add_modifier(Modifier::BOLD),
            )),
    )
}

#[derive(Clone, Copy)]
pub enum TuiAction {
    None,
    Quit,
    ToggleDebug,
}

/// Non-blocking poll for a single keypress.
pub fn poll_action() -> io::Result<TuiAction> {
    if event::poll(Duration::from_millis(0))?
        && let Event::Key(KeyEvent {
            code, modifiers, ..
        }) = event::read()?
    {
        if matches!(code, KeyCode::Char('q') | KeyCode::Esc)
            || (modifiers.contains(KeyModifiers::CONTROL) && matches!(code, KeyCode::Char('c')))
        {
            return Ok(TuiAction::Quit);
        }
        if matches!(code, KeyCode::Char('d')) {
            return Ok(TuiAction::ToggleDebug);
        }
    }
    Ok(TuiAction::None)
}
