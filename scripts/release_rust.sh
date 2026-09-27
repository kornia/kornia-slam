#!/usr/bin/env bash
#
# Publish the workspace's library crates to crates.io, dependencies first.
#
# Default: plan mode — prints what would be published and dry-runs the
# crates whose dependencies are all already on crates.io. A crate that
# depends on a not-yet-published workspace crate cannot be dry-run; Cargo
# resolves it from the registry.
#
# --execute publishes for real and needs CARGO_REGISTRY_TOKEN.
#
# Idempotent: a crate whose local version is already on crates.io is
# skipped, so a failed run can be resumed.

set -euo pipefail

MODE="plan"
SLEEP_SECS=20
USER_AGENT="kornia-slam release script (https://github.com/kornia/kornia-slam)"

for arg in "$@"; do
  case "$arg" in
    --execute) MODE="execute" ;;
    --help|-h) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# kornia-slam depends on kornia-sensors.
PUBLISH_ORDER=(kornia-sensors kornia-slam)

cd "$(dirname "$0")/.."
python3 scripts/check_version_pins.py

local_version() {
  cargo metadata --format-version 1 --no-deps \
    | python3 -c "import sys,json; print(next(p['version'] for p in json.load(sys.stdin)['packages'] if p['name'] == '$1'))"
}

# Prints the published version, NOT-PUBLISHED on a 404, or ERROR-HTTP-<code>
# on anything else so a network failure can never be mistaken for a match.
remote_version() {
  local crate="$1" response code
  for _ in 1 2 3; do
    if response=$(curl -fsS -A "$USER_AGENT" "https://crates.io/api/v1/crates/$crate" 2>/dev/null); then
      python3 -c "import sys,json; print(json.load(sys.stdin)['crate']['max_version'])" <<<"$response"
      return 0
    fi
    sleep 2
  done
  code=$(curl -s -o /dev/null -w "%{http_code}" -A "$USER_AGENT" "https://crates.io/api/v1/crates/$crate" || echo 000)
  if [[ "$code" == "404" ]]; then echo "NOT-PUBLISHED"; else echo "ERROR-HTTP-$code"; fi
}

workspace_deps() {
  cargo metadata --format-version 1 --no-deps | python3 -c "
import sys, json
pkgs = json.load(sys.stdin)['packages']
names = {p['name'] for p in pkgs}
crate = next(p for p in pkgs if p['name'] == '$1')
print(' '.join(d['name'] for d in crate['dependencies'] if d['name'] in names and d['kind'] is None))
"
}

# True when every workspace crate the given crate depends on is on crates.io.
deps_published() {
  local dep
  for dep in $(workspace_deps "$1"); do
    [[ "$(remote_version "$dep")" == "$(local_version "$dep")" ]] || return 1
  done
}

if [[ "$MODE" == "plan" ]]; then
  echo "==> PLAN — publish order: ${PUBLISH_ORDER[*]}"
  for crate in "${PUBLISH_ORDER[@]}"; do
    local_v=$(local_version "$crate")
    remote_v=$(remote_version "$crate")
    if [[ "$local_v" == "$remote_v" ]]; then
      echo "    $crate: $remote_v already published, would skip"
      continue
    fi
    echo "    $crate: $remote_v -> $local_v"
    if deps_published "$crate"; then
      cargo publish -p "$crate" --dry-run
    else
      echo "    (dry-run skipped: a workspace dependency is not on crates.io yet)"
    fi
  done
  echo "==> Plan complete. Re-run with --execute to publish."
  exit 0
fi

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  echo "ERROR: CARGO_REGISTRY_TOKEN is not set." >&2
  exit 1
fi

published=0
for crate in "${PUBLISH_ORDER[@]}"; do
  local_v=$(local_version "$crate")
  remote_v=$(remote_version "$crate")
  if [[ "$local_v" == "$remote_v" ]]; then
    echo "==> $crate: $remote_v already on crates.io, skipping"
    continue
  fi
  echo "==> $crate: $remote_v -> $local_v"
  cargo publish -p "$crate"
  published=$((published + 1))
  if [[ "$crate" != "${PUBLISH_ORDER[-1]}" ]]; then
    echo "    waiting ${SLEEP_SECS}s for the crates.io index"
    sleep "$SLEEP_SECS"
  fi
done
echo "==> Done — published $published crate(s)"
