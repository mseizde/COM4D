"""External video super-resolution adapters."""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def run_rvrt_super_resolution(frames_dir: Path, *, rvrt_repo: Path,
                              python_executable: str, task: str,
                              tile: Sequence[int],
                              tile_overlap: Sequence[int]) -> list[Image.Image]:
    """Run the official RVRT CLI and return detached, ordered RGB frames."""
    frames_dir = frames_dir.expanduser().resolve()
    rvrt_repo = rvrt_repo.expanduser().resolve()
    script = rvrt_repo / "main_test_rvrt.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"RVRT entry point not found: {script}. "
            "Clone https://github.com/JingyunLiang/RVRT first."
        )
    if len(tile) != 3 or len(tile_overlap) != 3:
        raise ValueError("RVRT tile and tile overlap must each contain three integers")
    sources = [path for path in sorted(frames_dir.iterdir())
               if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not sources:
        raise ValueError(f"No supported images found in {frames_dir}")

    with tempfile.TemporaryDirectory(prefix="com4d_rvrt_") as tmp:
        work_dir = Path(tmp)
        sequence_dir = work_dir / "lq" / "com4d"
        sequence_dir.mkdir(parents=True)
        for index, source in enumerate(sources):
            target = sequence_dir / f"{index:08d}{source.suffix.lower()}"
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        model_zoo = rvrt_repo / "model_zoo"
        if model_zoo.exists():
            (work_dir / "model_zoo").symlink_to(model_zoo, target_is_directory=True)
        command = [
            python_executable, str(script), "--task", task,
            "--folder_lq", str(work_dir / "lq"), "--tile",
            *(str(int(value)) for value in tile), "--tile_overlap",
            *(str(int(value)) for value in tile_overlap),
            "--save_result",
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        print("Running RVRT:", " ".join(command))
        subprocess.run(command, cwd=work_dir, env=env, check=True)
        results_dir = work_dir / "results"
        outputs = sorted(path for path in results_dir.rglob("*")
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(outputs) != len(sources):
            raise RuntimeError(
                f"RVRT produced {len(outputs)} frames below {results_dir}; "
                f"expected {len(sources)}"
            )
        refined = []
        for path in outputs:
            with Image.open(path) as image:
                refined.append(image.convert("RGB").copy())
        return refined
