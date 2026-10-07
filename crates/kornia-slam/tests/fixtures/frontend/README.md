# Frontend fixtures

FNV-1a 64-bit hashes of frontend outputs recorded from the app's frontend before
it moved into `SlamSystem` (commit `7fd563c`, rustc 1.98.1, kornia-rs 0.2.0). The
tests in `src/frontend/tests.rs` regenerate the same inputs and must
reproduce these values exactly.

- `frontend.txt`: ORB features (1000-keypoint detector), stereo matches and
  keypoint colors for a deterministic 376x240 texture of 4x4 blocks of
  splitmix-style noise, with the right view shifted by 8 pixels; rig with
  `fx = 435` and a 0.11 m baseline.
- `hilti.txt`: a 40x40 keypoint grid over the 1472x1440 Hilti fisheye image,
  mapped into the virtual pinhole with the 88° incidence cap. Each point
  carries an index-derived orientation, descriptor and octave, so the hash
  also checks that dropped points take their attributes with them.

Regenerating them from new code would defeat their purpose: a change that
alters these values changes frontend output and needs an end-to-end
evaluation instead.
