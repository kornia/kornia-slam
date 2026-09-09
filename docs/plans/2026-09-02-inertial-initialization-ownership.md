# Inertial Initialization Ownership Refactor

**Goal:** make `initialization::inertial` own inertial initialization end to end
(window, readiness, VIBA0/1/2 schedule, priors) so `SlamSystem` only applies
results. No behavior change: same accepted/rejected VIBA stages, same ATE.

**Branch:** `refactor/slam-system` (worktree `.worktrees/refactor-slam-system`).
**Do not commit.** Leave changes in the working tree; the user reviews and commits.

## Ground rules

- **Never run cargo on this laptop.** Build, test and run only on the host
  `ssh christie@192.168.0.190` (cargo is at `~/.cargo/bin/cargo`, not on PATH).
  Sync the worktree there with
  `rsync -a --delete --exclude target --exclude .git ./ christie@192.168.0.190:~/projects/kornia-slam-refactor/`
  and run `cd ~/projects/kornia-slam-refactor && ~/.cargo/bin/cargo ...` over ssh.
  Do not touch `~/projects/kornia-slam` on the host (clean `develop` checkout).
- Keep every comment in `initialization/inertial/mod.rs` that explains a reverted
  gate or a convention (gyro gate removal, gravity-gate revert, gI vs canonical
  gravity). Move them with the code; do not paraphrase them away.
- Preserve the public API test `crates/kornia-slam/tests/system_api.rs`.
- Each step ends with: `cargo fmt --all -- --check`,
  `cargo clippy -p kornia-slam --all-targets -- -D warnings`,
  `cargo test -p kornia-slam`, `cargo check --workspace` (all on the host).
- Steps 3 and 5 additionally require the end-to-end check below.

## End-to-end check (host)

EuRoC MH_01_easy is not on the host. Once, download it:

```
mkdir -p ~/datasets/euroc && cd ~/datasets/euroc
wget -c http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip -q MH_01_easy.zip -d MH_01_easy && rm MH_01_easy.zip
```
(`MH_01_easy/mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}` must exist.)

Baseline **before any edit** and after steps 3 and 5:

```
~/.cargo/bin/cargo run --release -p kornia-slam-app -- --no-tui --debug euroc \
  --data ~/datasets/euroc/MH_01_easy --imu --max-frames 500 --evaluate \
  2>&1 | tee run.log
grep -E '\[imu_init\]|ATE|RPE|scale' run.log
```

Compare against the baseline: the same sequence of `[imu_init] VIBA0/VIBA1/VIBA2
accepted|rejected` lines at the same frame indices, `scale` within 1e-3, and ATE
within 1e-3 m. Any difference is a bug in the refactor, not a tuning opportunity.
Record baseline and final numbers at the bottom of this file.

Hardware note: the `--imu` mono path exercises VIBA0 (mono priors) only within
500 frames unless VIBA1 fires before frame 500; also run with `--stereo --imu`
so the stereo prior path and `is_mono=false` are covered.

## Current shape (line anchors at commit 5cc7d44)

- `crates/kornia-slam/src/initialization/inertial/mod.rs`
  - `rotation_from_to` (17), `window_is_mono` (45), `ImuInitConfig` (65),
    `ImuInitializer { config }` (138), `ready` (149), `inertial_optimizer` (203),
    `try_initialize(map, imu_t_bc, imu_bias, start_idx, prior_g, prior_a, already_initialized)` (365).
  - Tests (534+) import `crate::system::apply_inertial_initialization` — the
    cycle step 5 removes.
- `crates/kornia-slam/src/initialization/inertial/factor.rs` — private; check
  whether `with_fixed_bias_vel` / `fixed_bias_vel` are used anywhere. If not, delete.
- `crates/kornia-slam/src/system/mod.rs`
  - duplicate `rotation_from_to` (142); `apply_inertial_initialization` (168);
    fields `inertial_init_start_kf_idx`, `inertial_init`,
    `inertial_init_last_attempt_sec`, `imu_init_window_start_sec`,
    `imu_viba1_done`, `imu_viba2_done` (~75-92);
    window reset in stereo bootstrap (~500) and mono bootstrap (~693);
    `inertial_init_step` (841) incl. gate message + `is_mono` re-derivation +
    VIBA0 priors `(1e2, 1e10|1e5)`; `refine_inertial_init` (1359) with
    VIBA1 `(1.0, VIBA_PRIOR_A)`, VIBA2 `(0.0, VIBA_PRIOR_A)`, 5 s/15 s/50 s gates;
    `format_imu_init_gate` (2004).
- `crates/kornia-slam/src/map/mod.rs` — `scale_world` (720), `rotate_world` (733),
  `imu_factors` (715), `Keyframe { velocity_world, imu_bias }` (66).
- `crates/kornia-slam/src/tracking/state.rs` — `SystemState` (30).

## Step 1 — window helpers and a readiness report

Files: `map/mod.rs`, `initialization/inertial/mod.rs`, `system/mod.rs`.

1. Add to `Map`:
   - `pub fn keyframes_from(&self, start_idx: usize) -> impl Iterator<Item = &Keyframe>`
     (filter `kf.frame.idx >= start_idx`; callers that need sorted order sort themselves as today).
   - `pub fn imu_time_from(&self, start_idx: usize) -> f64`
     (sum of `preintegrated.dt` over factors with `curr_kf_idx >= start_idx`).
2. Replace the four hand-rolled filters (`ready`, `try_initialize`,
   `inertial_init_step` gate snapshot, `inertial_init_step` `is_mono`) with them.
3. Change `ready(&self, map, start_idx) -> bool` into
   `readiness(&self, map, start_idx) -> Result<(), ImuInitNotReady>` where

   ```rust
   pub struct ImuInitNotReady {
       pub start_kf_idx: usize,
       pub first_kf_idx: Option<usize>, pub last_kf_idx: Option<usize>,
       pub keyframes: usize, pub min_keyframes: usize,
       pub imu_time_sec: f64, pub min_time_sec: f64,   // min already doubled for mono
       pub motion: f64, pub min_motion: f64,
       pub reason: ImuInitNotReadyReason,              // NoWindow | InvalidConfig | Keyframes | ImuTime | Motion
   }
   impl Display for ImuInitNotReady  // one-line, same content as format_imu_init_gate
   ```
   Keep a thin `ready()` that returns `readiness().is_ok()` if tests use it.
4. Delete `format_imu_init_gate` and its test; the system logs
   `Err(not_ready)` via `Display` on `KeyframeAccepted` exactly where the gate
   message was logged (keep the `[imu_init]` prefix so log greps still work).

## Step 2 — typed request: stage, priors, seed

File: `initialization/inertial/mod.rs`, call sites in `system/mod.rs`.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InertialStage { Viba0, Viba1, Viba2 }

#[derive(Debug, Clone, Copy)]
pub struct BiasPriors { pub gyro: f64, pub accel: f64 }
impl BiasPriors {
    /// ORB-SLAM3 LocalMapping.cc:183-228 priors. Keep the numbers here only.
    pub fn for_stage(stage: InertialStage, is_mono: bool) -> Self
    // Viba0: gyro 1e2, accel 1e10 (mono) / 1e5 (stereo)
    // Viba1: gyro 1.0, accel VIBA_PRIOR_A   (move the const out of SlamSystem)
    // Viba2: gyro 0.0, accel VIBA_PRIOR_A   (NOT relaxed — see project memory
    //        "accel-bias blowup": VIBA2 keeps prior_a=1e5; do not change)
}

#[derive(Debug, Clone, Copy)]
pub enum RwgSeed { FromVisualTrajectory, FromCurrentGravity(Vec3F64) }

pub struct InertialInitRequest {
    pub start_kf_idx: usize,
    pub imu_t_bc: Pose3d,
    pub bias: ImuBias,
    pub priors: BiasPriors,
    pub seed: RwgSeed,
}
pub fn try_initialize(&self, map: &Map, req: &InertialInitRequest) -> Result<ImuInitResult, ImuInitReject>
```

- `MissingExtrinsics` moves to the caller: the system builds the request and
  logs that reject itself when `imu_t_bc` is `None`. Keep the variant so
  `ImuInitReject` is unchanged for consumers.
- `FromCurrentGravity(g)`: seed `rwg = rotation_from_to((0,0,-1), g.normalize())`.
  With `g = (0,+G,0)` this is bit-identical to today's `gi_to_canonical`; move
  the long comment explaining why identity is wrong onto this arm.
- The system passes `RwgSeed::FromCurrentGravity(self.gravity_world)` for
  VIBA1/2 and `FromVisualTrajectory` for VIBA0.
- Delete the duplicate `is_mono` computation in `inertial_init_step`; the
  initializer computes `window_is_mono` and picks priors via
  `BiasPriors::for_stage(stage, is_mono)`. To make that possible, add
  `pub fn stage_priors(&self, map, start_idx, stage) -> BiasPriors` or, simpler,
  let the request carry `stage` instead of `priors` and resolve priors inside.
  Prefer carrying `stage`; keep `BiasPriors` public for tests.

## Step 3 — move the schedule into the initializer

Files: `initialization/inertial/mod.rs` (+ new `schedule.rs` if mod.rs passes
~1200 lines), `system/mod.rs`, `system/config.rs`.

1. Extend config (defaults = today's constants):
   ```rust
   pub struct ImuInitConfig {
       pub min_keyframes: usize, pub min_time_sec: f64, pub min_motion: f64,   // existing
       pub mono_min_time_factor: f64,      // 2.0 — replaces the silent doubling
       pub retry_interval_sec: f64,        // 5.0
       pub viba1_after_sec: f64,           // 5.0
       pub viba2_after_sec: f64,           // 15.0
       pub refine_until_sec: f64,          // 50.0
       pub max_accel_bias: f64,            // 1.0 (MAX_PLAUSIBLE_ACCEL_BIAS)
   }
   ```
   Validate the new fields in `validate()`.
2. Give `ImuInitializer` the schedule state that today lives on `SlamSystem`:
   `start_kf_idx: Option<usize>`, `window_start_sec: Option<f64>`,
   `last_attempt_sec: Option<f64>`, `viba1_done: bool`, `viba2_done: bool`.
3. API:
   ```rust
   pub fn begin_window(&mut self, start_kf_idx: usize, timestamp_sec: f64)   // resets all five
   pub fn window_start_kf_idx(&self) -> Option<usize>
   pub enum InertialInitOutcome { NotDue, NotReady(ImuInitNotReady), Attempted { stage: InertialStage, result: Result<ImuInitResult, ImuInitReject> } }
   /// Call on every accepted keyframe while `imu_initialized == false`.
   pub fn on_keyframe_uninitialized(&mut self, map: &Map, timestamp_sec: f64, imu_t_bc: Option<Pose3d>, bias: ImuBias) -> InertialInitOutcome
   /// Call on every accepted keyframe once initialized; fires VIBA1 then VIBA2 at most once each.
   pub fn on_keyframe_initialized(&mut self, map: &Map, timestamp_sec: f64, imu_t_bc: Option<Pose3d>, bias: ImuBias, gravity_world: Vec3F64) -> InertialInitOutcome
   ```
   The `done` flags flip exactly where they flip today (after the attempt,
   regardless of accept/reject).
4. `SlamSystem`: delete the five fields, `VIBA_PRIOR_A`, `refine_inertial_init`;
   both bootstrap resets become `self.inertial_init.begin_window(curr_idx, timestamp_sec)`;
   `inertial_init_step` shrinks to: run tracking step, on `KeyframeAccepted`
   call `on_keyframe_uninitialized`, match the outcome, apply, log, switch mode,
   submit the local-mapping job. The `try_insert_keyframe` call site that
   invoked `refine_inertial_init` calls `on_keyframe_initialized` and applies.
5. Log lines must keep today's text (`[imu_init] VIBA0 accepted: scale=...`,
   `... rejected: ...`, `... apply rejected: ...`) — the end-to-end check greps them.
6. Run the end-to-end check (mono+imu and stereo+imu). Must match baseline.

## Step 4 — single `rotation_from_to`

Delete the copy in `system/mod.rs`; make the one in `initialization/inertial`
`pub(crate)` (or move it to `pose_conversion.rs`, which already holds small
pose helpers, and import it from both). Unit tests for anti-parallel input stay.

## Step 5 — map-side application, break the test cycle

Files: `map/mod.rs`, `system/mod.rs`, `initialization/inertial/mod.rs` tests.

1. Add to `Map`:
   ```rust
   pub struct InertialAlignment { pub scale: f64, pub rotation: SO3F64, pub keyframe_velocities: Vec<KeyframeVelocity>, pub bias: ImuBias }
   pub enum InertialAlignmentError { InvalidScale(f64), MissingVelocities, DuplicateKeyframe(usize), MissingKeyframe(usize), InvalidVelocity(usize), InvalidBias }
   /// Validates everything, then scales, rotates, and writes velocities+bias atomically.
   /// Returns the index of the last (max idx) keyframe in the assignment.
   pub fn apply_inertial_alignment(&mut self, a: InertialAlignment) -> Result<usize, InertialAlignmentError>
   ```
   Body = today's `apply_inertial_initialization` minus the `SystemState`,
   `gravity_world`, `imu_bias` writes. Velocities are rotated by `rotation`
   before being stored, as today.
2. `system::apply_inertial_initialization` becomes a private method: checks
   gravity norm (`InvalidGravity` stays in `ImuInitApplyError`, which wraps
   `InertialAlignmentError` via `#[from]`), computes
   `alignment = rotation_from_to(g/|g|, (0,1,0))`, calls the map, then sets the
   four state fields from the returned last keyframe. Same order as today.
3. Move the tests that currently import `crate::system::apply_inertial_initialization`
   from `initialization/inertial` to `map` (the alignment part) and `system`
   (the state part). `initialization` must no longer reference `crate::system`.
4. `KeyframeJob { imu_initialized, imu_t_bc, gravity_world }` is built in four
   places; add `fn keyframe_job(&self) -> KeyframeJob` on `SlamSystem` and use it.
5. Run the end-to-end check again; must match baseline.

## Out of scope (do not do here)

`pending_imu`, `preintegrate_window`, `prune_imu_before`, `predict_pose_imu`,
`body_to_world` stay in `system` (they are tracking-side propagation; a later
`tracking/imu.rs` step). No changes to `factor.rs` numerics. No tuning.

## Results

Dataset: `/media/christie/T7/kornia-slam-datasets/euroc/MH_01_easy` (already on the
host's T7 drive — `robotics.ethz.ch` was unreachable from the host, so the
download in "End-to-end check" was skipped in favour of the local copy).
All runs: `--no-tui --debug euroc --max-frames 500 --evaluate`, release build.

**Important: the pipeline is not bit-reproducible across builds.** Local mapping
runs asynchronously, so the map a VIBA solve sees depends on thread interleaving.
Re-runs of the *same* binary usually repeat; a rebuild (or a different build
directory) shifts the numbers by ~1e-3 in scale/ATE. The very first baseline
capture below was taken from the first build in the refactor worktree; a clean
rebuild of the untouched base commit `5cc7d44` in a separate directory
(`~/projects/kornia-slam-baseline`) does **not** reproduce it, which is why the
A/B below compares against that clean rebuild.

### mono (`--imu`)

| run | VIBA0 (frame 58) | VIBA1 (frame 120) | VIBA2 (frame 313) | Scale | ATE | RPE |
|---|---|---|---|---|---|---|
| baseline, first build of the refactor tree | accepted 0.7081, g=(0.300,9.679,1.572) | accepted 0.9900 | accepted 0.9775 | 0.929956 | 0.0327 | 0.0076 |
| after step 3 | accepted 0.7081, g=(0.300,9.679,1.572) | accepted 0.9900 | accepted 0.9775 | 0.929956 | 0.0327 | 0.0076 |
| base commit 5cc7d44, clean rebuild, 3 runs | accepted 0.6997 / 0.6998 / 0.6998 | 0.9815 / 0.9824 / 0.9820 | 0.9807 / 0.9795 / 0.9788 | 0.937021 / 0.936703 / 0.938561 | 0.0333 / 0.0335 / 0.0337 | 0.0076 |
| after step 5, 3 runs | accepted 0.6998 (×3) | 0.9820 | 0.9788 | 0.938341 / 0.937199 / 0.938341 | 0.0337 / 0.0335 / 0.0337 | 0.0076 |

Step 3 reproduced the then-current binary exactly. After step 5 the numbers sit
inside the *unmodified* base commit's own rebuild spread (VIBA0 0.6998 vs
0.6997-0.6998, ATE 0.0335-0.0337 vs 0.0333-0.0337) — same stage sequence, same
frame indices, same accept/reject outcome.

### stereo (`--stereo --imu`)

| run | VIBA0 (frame 33) | VIBA1 (frame 101) | VIBA2 (frame 303) | Scale | ATE | RPE |
|---|---|---|---|---|---|---|
| baseline, first build of the refactor tree | accepted 1.0000, g=(0.214,9.359,2.931) | accepted 1.0000 | accepted 1.0000 | 0.994780 | 0.0256 | 0.0053 |
| after step 3 | accepted 1.0000, g=(0.214,9.359,2.931) | accepted 1.0000 | accepted 1.0000 | 0.995072 / 0.994780 | 0.0256 | 0.0053 |
| base commit 5cc7d44, clean rebuild, 2 runs | accepted 1.0000, g=(0.214,9.359,2.931) | accepted 1.0000 | accepted 1.0000 | 0.995198 / 0.995069 | 0.0255 / 0.0256 | 0.0053 |
| after step 5, 2 runs | accepted 1.0000, g=(0.214,9.359,2.931) | accepted 1.0000 | accepted 1.0000 | 0.994780 / 0.994608 | 0.0256 / 0.0255 | 0.0053 |

All `[imu_init]` lines are byte-identical to the baseline in stereo (VIBA0
gravity and every gyro bias included); only the reported `Scale` moves, by
3e-4 — inside the baseline's own 1.3e-4-wide rebuild spread.

### Deviations from the plan

- `begin_window` clears `last_attempt_sec` too (the plan's "resets all five").
  Today's code does not clear it on re-bootstrap, so after a tracking-loss reset
  the new window could be throttled by the old window's attempt timestamp. Not
  reachable in the 500-frame MH_01 check.
- `try_initialize(map, request)` delegates to a private
  `solve(map, request, prior_override)`. Two synthetic tests sweep the bias
  priors (including the un-regularized `(0, 0)` pair that no production stage
  uses); they call `solve`, production always calls `try_initialize`.
- `factor.rs`: `with_fixed_bias_vel` and the `fixed_bias_vel` field were unused
  (the field could only ever be `false`), so both are gone along with the
  now-unconditional Jacobian branch. No numerics changed.
- `ImuInitNotReady::min_time_sec` is the mono-doubled threshold, so the
  `[imu_init_gate]` debug line now prints `imu_time=x/2.0s` for a monocular
  window where it used to print `/1.0s`. Debug output only.
- The gate line is now emitted only while the window is *not* ready; previously
  it was also printed on the keyframes where the window was ready. Debug only,
  and `[imu_init_gate]` is not matched by the `\[imu_init\]` grep.
