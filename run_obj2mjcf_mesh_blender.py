#!/usr/bin/env python3
"""Run obj2mjcf for every immediate subfolder in a mesh directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MESH_ROOT = Path("~/shared_data/mesh_blender")
DEFAULT_FAILED_LOG = Path("obj2mjcf_failed_objects.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run obj2mjcf on each immediate subfolder of a mesh directory."
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=DEFAULT_MESH_ROOT,
        help=f"Directory containing object subfolders (default: {DEFAULT_MESH_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Deprecated: this is now the default behavior.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first obj2mjcf failure.",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=DEFAULT_FAILED_LOG,
        help=f"Text file for failed object folders (default: {DEFAULT_FAILED_LOG})",
    )
    parser.add_argument(
        "--output-folder",
        default="{object_name}",
        help=(
            "Expected output folder name used for skipping completed objects. "
            "Use {object_name} for the current subfolder name."
        ),
    )
    return parser.parse_args()


def object_dirs(mesh_root: Path) -> list[Path]:
    return sorted(path for path in mesh_root.iterdir() if path.is_dir())


def expected_output_dir(folder: Path, output_folder: str) -> Path:
    return folder / output_folder.format(object_name=folder.name)


def main() -> int:
    args = parse_args()
    mesh_root = args.mesh_root.expanduser().resolve()
    failed_log = args.failed_log.expanduser().resolve()

    if not mesh_root.is_dir():
        print(f"Mesh root does not exist or is not a directory: {mesh_root}", file=sys.stderr)
        return 1

    folders = object_dirs(mesh_root)
    if not folders:
        print(f"No subfolders found in: {mesh_root}", file=sys.stderr)
        return 1

    command = [
        "obj2mjcf",
        "--obj-dir",
        ".",
        "--save-mjcf",
        "--decompose",
        "--overwrite",
        "--coacd-args.preprocess-resolution",
        "90",
        "--coacd-args.threshold",
        "0.03",
    ]

    failures: list[tuple[Path, int]] = []
    skipped = 0
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        failed_log.write_text("", encoding="utf-8")

    for folder in folders:
        output_dir = expected_output_dir(folder, args.output_folder)
        if output_dir.is_dir():
            skipped += 1
            print(f"[{folder.name}] skipped; output folder exists: {output_dir}", flush=True)
            continue

        print(f"[{folder.name}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=folder)
        if result.returncode != 0:
            failures.append((folder, result.returncode))
            with failed_log.open("a", encoding="utf-8") as file:
                file.write(f"{folder}\texit code {result.returncode}\n")
            if args.stop_on_error:
                break

    print(f"\nSkipped existing outputs: {skipped}")
    if failures:
        print(f"Failed object list written to: {failed_log}", file=sys.stderr)
        print("\nFailures:", file=sys.stderr)
        for folder, returncode in failures:
            print(f"  {folder}: exit code {returncode}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
