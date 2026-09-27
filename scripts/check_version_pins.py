#!/usr/bin/env python3
"""Fail if a workspace crate's pin in [workspace.dependencies] disagrees with
workspace.package.version.

Cargo cannot inherit the version into those pins, and locally the `path`
wins, so a stale pin builds fine and only ships a wrong requirement when
the crate is published.
"""
import sys
import tomllib
from pathlib import Path

manifest = tomllib.loads((Path(__file__).resolve().parent.parent / "Cargo.toml").read_text())
workspace = manifest["workspace"]
version = workspace["package"]["version"]

stale = [
    f"  {name}: {dep.get('version', '<missing>')}"
    for name, dep in workspace["dependencies"].items()
    if isinstance(dep, dict) and dep.get("path", "").startswith("crates/")
    and dep.get("version") != version
]

if stale:
    print(f"workspace crate pins out of sync with workspace.package.version ({version}):")
    print("\n".join(stale))
    sys.exit(1)
print(f"version pins OK ({version})")
