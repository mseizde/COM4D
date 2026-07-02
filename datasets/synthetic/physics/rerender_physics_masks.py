#!/usr/bin/env python3
"""Rerender exact visible/amodal masks for retained physics raw samples.

The original Blender command is recovered from each pipeline.log so camera
parameters match the existing RGB frames. Simulation, RGB, point sampling,
transforms, and mesh export are not rerun.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shlex
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def original_blender_command(log_path: Path) -> list[str]:
    lines = log_path.read_text(errors="replace").splitlines()
    candidates = [
        line[2:] for line in lines
        if line.startswith("+ ") and "render_physics_outputs.py" in line
    ]
    if not candidates:
        raise ValueError(f"No renderer command found in {log_path}")
    return shlex.split(candidates[-1])


def remove_flag(command: list[str], flag: str, values: int = 0) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + 1 + values]


def mask_command(sample_dir: Path, overwrite: bool) -> list[str] | None:
    expected = sample_dir / "masks_amodal"
    if expected.is_dir() and not overwrite:
        metadata = json.loads((sample_dir / "physics_metadata.json").read_text())
        frame_count = int(metadata.get("num_frames", 0) or len(metadata.get("frames", [])))
        object_count = len(metadata.get("render_objects", []))
        expected_count = frame_count * object_count
        pngs = list(expected.rglob("*.png"))
        if expected_count > 0 and len(pngs) >= expected_count:
            return None
    command = original_blender_command(sample_dir / "pipeline.log")
    remove_flag(command, "--skip-masks")
    remove_flag(command, "--engine", values=1)
    for flag in (
        "--skip-rgb",
        "--write-amodal-masks",
        "--skip-transforms",
        "--skip-canonical-meshes",
        "--skip-camera-metadata",
    ):
        if flag not in command:
            command.append(flag)
    command.extend(["--engine", "BLENDER_EEVEE", "--mask-mode", "material"])
    return command


def run_sample(sample_dir: Path, overwrite: bool, dry_run: bool) -> tuple[str, str]:
    command = mask_command(sample_dir, overwrite)
    if command is None:
        return sample_dir.name, "skip"
    print("[command]", shlex.join(command), flush=True)
    if dry_run:
        return sample_dir.name, "dry-run"
    log_path = sample_dir / "mask_rerender.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(f"{sample_dir.name}: Blender failed; see {log_path}")
    return sample_dir.name, "rendered"


def main() -> None:
    args = parse_args()
    root = args.raw_root.expanduser().resolve()
    samples = (
        [root / name for name in args.sample]
        if args.sample
        else sorted(path for path in root.iterdir() if path.is_dir())
    )
    samples = [
        path for path in samples
        if (path / "physics_metadata.json").is_file() and (path / "pipeline.log").is_file()
    ]
    if not samples:
        raise SystemExit(f"No retained raw samples with pipeline.log under {root}")

    counts = {"rendered": 0, "skip": 0, "dry-run": 0}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_sample, sample, args.overwrite, args.dry_run): sample
            for sample in samples
        }
        for future in as_completed(futures):
            name, status = future.result()
            counts[status] += 1
            print(f"[{status}] {name}", flush=True)
    print(counts)


if __name__ == "__main__":
    main()

