#!/usr/bin/env python3
"""Bumps the release version in both places that have to agree on it.

`pyproject.toml` is what PyPI sees, and `uv.lock` carries a copy of it in the
project's own entry -- so bumping only the first leaves `uv sync --locked` (and CI
with it) rejecting the very commit that was released. Editing them by hand is how
they drift, so the auto-release job calls this instead.

    tools/bump_version.py patch          # 0.1.0 -> 0.1.1
    tools/bump_version.py minor          # 0.1.0 -> 0.2.0
    tools/bump_version.py major          # 0.1.0 -> 1.0.0
    tools/bump_version.py 1.4.2          # set exactly

The single line printed to stdout is the new version, which the workflow reads back
to form the tag. Nothing else is printed there, so it can be captured directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"


def project_name() -> str:
    """The distribution name, which is the key `uv.lock` files the version under."""
    match = re.search(r'(?m)^\[project\][^\[]*?^name = "([^"]+)"', PYPROJECT.read_text())
    if match is None:
        raise SystemExit("could not find [project] name in pyproject.toml")
    # uv normalises names in the lock file the way PEP 503 does.
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def current_version() -> str:
    match = re.search(r'(?m)^\[project\][^\[]*?^version = "([^"]+)"', PYPROJECT.read_text())
    if match is None:
        raise SystemExit("could not find [project] version in pyproject.toml")
    return match.group(1)


def next_version(current: str, spec: str) -> str:
    if spec not in {"major", "minor", "patch"}:
        if not re.fullmatch(r"\d+\.\d+\.\d+", spec):
            raise SystemExit(f"expected major|minor|patch or an X.Y.Z version, got {spec!r}")
        return spec
    major, minor, patch = (int(part) for part in current.split("."))
    if spec == "major":
        return f"{major + 1}.0.0"
    if spec == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def set_pyproject(version: str) -> None:
    text, count = re.subn(
        r'(?m)(^\[project\][^\[]*?^version = )"[^"]+"',
        rf'\g<1>"{version}"',
        PYPROJECT.read_text(),
        count=1,
    )
    if count != 1:
        raise SystemExit("failed to rewrite the version in pyproject.toml")
    PYPROJECT.write_text(text)


def set_uv_lock(version: str) -> None:
    """Rewrite only the project's own entry, never a dependency that shares a version."""
    name = project_name()
    text, count = re.subn(
        rf'(\[\[package\]\]\nname = "{re.escape(name)}"\nversion = )"[^"]+"',
        rf'\g<1>"{version}"',
        UV_LOCK.read_text(),
        count=1,
    )
    if count != 1:
        raise SystemExit(f"failed to rewrite the {name} version in uv.lock")
    UV_LOCK.write_text(text)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    version = next_version(current_version(), sys.argv[1])
    set_pyproject(version)
    set_uv_lock(version)
    print(version)


if __name__ == "__main__":
    main()
