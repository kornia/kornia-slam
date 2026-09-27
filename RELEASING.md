# Releasing

kornia-slam publishes two things: the `kornia-sensors` and `kornia-slam` crates to
crates.io, and prebuilt `kornia-slam` CLI binaries attached to a GitHub Release.
`kornia-slam-app` is `publish = false`.

## One-time setup

1. A crates.io API token with the `publish-new` and `publish-update` scopes, stored
   as the `CARGO_REGISTRY_TOKEN` secret of a `crates-io` environment
   (Settings → Environments). Restrict the environment to `main` and `v*` tags and
   add required reviewers.
2. After the first publish, add co-owners on crates.io
   (`cargo owner --add <user> kornia-slam`), so a release never depends on one account.

## Cutting a release

1. **Bump the version.** Set `[workspace.package].version` in `Cargo.toml`, and the
   `kornia-sensors` / `kornia-slam` pins in `[workspace.dependencies]` to the same
   value. `scripts/check_version_pins.py` (also run in CI) fails if they disagree.
2. **Update `CHANGELOG.md`.** Move the `[Unreleased]` items into a dated
   `[X.Y.Z]` section and fix the link references at the bottom.
3. **Merge** that PR to `main` with CI green.
4. **Tag** the merge commit and push the tag:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
   ```
   The `Binary Release` workflow builds linux x86_64 and aarch64 binaries and opens a
   draft GitHub Release with them.
5. **Publish the crates.** Run the `Rust Release` workflow from the Actions tab on the
   tag and type `publish` to confirm. It runs `scripts/release_rust.sh`, which publishes
   `kornia-sensors` before `kornia-slam` and skips anything already on crates.io, so a
   failed run can be re-run.
6. **Publish the GitHub Release.** Review the draft, paste the CHANGELOG section into
   the notes, and publish it.

To preview step 5 locally, run `./scripts/release_rust.sh`. It only plans and dry-runs;
`--execute` publishes.
