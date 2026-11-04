# Frontend fixtures

FNV-1a 64-bit hashes of frontend outputs recorded from the app's frontend before
it moved into `SlamSystem` (kornia-slam#149; rustc 1.98.1, kornia-rs 0.2.0). The
tests in `src/frontend/tests.rs` regenerate the same inputs and must
reproduce these values exactly.

- `frontend.txt`: ORB features (1000-keypoint budget, which the pyramid
  overshoots), stereo matches and keypoint colors for a deterministic 376x240
  texture of 4x4 blocks of splitmix-style noise, with the right view shifted by
  8 pixels; rig with `fx = 435` and a 0.11 m baseline.
- `orientations.txt`: the ORB orientations of that same frame, one per line in
  radians, compared within a tolerance instead of hashed. They come from
  `f32::atan2`, which calls the platform's libm, and libm versions differ in
  the last bit: the hash these values replaced failed on Ubuntu 26.04 and
  Windows. The descriptors steer by these angles too but round the rotated
  offsets to integers, so their hash still holds unless a libm change lands an
  offset on a rounding boundary.
- `hilti.txt`: a 40x40 keypoint grid over the 1472x1440 Hilti fisheye image,
  mapped into the virtual pinhole with the 88° incidence cap. Each point
  carries an index-derived orientation, descriptor and octave, so the hash
  also checks that dropped points take their attributes with them.

Regenerate them only together with an end-to-end EuRoC comparison: a change
that alters these values changes frontend output, including an upstream ORB
change in kornia-rs.
