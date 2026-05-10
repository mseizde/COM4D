from __future__ import annotations

import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


_CACHE: "LocalDatasetCache | None" = None
_CACHE_LOCK = threading.Lock()
_RECENT_FILE_GRACE_SECONDS = 600


def configure_dataset_cache(
    source_root: str | None,
    cache_root: str | None,
    *,
    enabled: bool,
    prefetch_window: int = 0,
    prefetch_workers: int = 2,
    max_cache_gb: float = 0.0,
) -> None:
    global _CACHE
    with _CACHE_LOCK:
        if not enabled or not source_root or not cache_root:
            _CACHE = None
            return
        _CACHE = LocalDatasetCache(
            source_root=source_root,
            cache_root=cache_root,
            prefetch_window=prefetch_window,
            prefetch_workers=prefetch_workers,
            max_cache_gb=max_cache_gb,
        )


def dataset_cache() -> "LocalDatasetCache | None":
    return _CACHE


def resolve_path(path: str) -> str:
    cache = dataset_cache()
    if cache is None:
        return path
    return cache.resolve(path)


def prefetch_paths(paths: Iterable[str]) -> None:
    cache = dataset_cache()
    if cache is not None:
        cache.prefetch(paths)


def extract_data_paths(data_config: dict, *, max_frame_paths: int = 8) -> list[str]:
    paths: list[str] = []
    if "frames" in data_config:
        frames = data_config["frames"]
        if max_frame_paths > 0 and len(frames) > max_frame_paths:
            step = max(1, len(frames) // max_frame_paths)
            frames = frames[::step][:max_frame_paths]
        for frame in frames:
            surface_path = frame.get("surface_path")
            image_path = frame.get("image_path")
            if surface_path:
                paths.append(surface_path)
            if image_path:
                paths.append(image_path)
    else:
        surface_path = data_config.get("surface_path")
        if surface_path:
            paths.append(surface_path)
        for surface_path in data_config.get("surface_paths", []):
            if surface_path:
                paths.append(surface_path)
        image_path = data_config.get("image_path")
        if image_path:
            paths.append(image_path)
        for aux_key in ("depth_path", "normal_path", "semantic_path", "geometry_metadata_path"):
            aux_path = data_config.get(aux_key)
            if aux_path:
                paths.append(aux_path)
    return paths


def prefetch_data_configs(data_configs: list[dict], start: int, window: int) -> None:
    if window <= 0:
        return
    paths: list[str] = []
    stop = min(len(data_configs), start + window)
    for idx in range(start, stop):
        config = data_configs[idx]
        if config:
            paths.extend(extract_data_paths(config))
    prefetch_paths(paths)


class LocalDatasetCache:
    def __init__(
        self,
        source_root: str,
        cache_root: str,
        *,
        prefetch_window: int = 0,
        prefetch_workers: int = 2,
        max_cache_gb: float = 0.0,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.prefetch_window = max(0, int(prefetch_window))
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.max_cache_bytes = int(float(max_cache_gb) * 1024**3) if max_cache_gb else 0
        self._executor: ThreadPoolExecutor | None = None
        self._submitted: set[str] = set()
        self._submitted_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._copies_since_cleanup = 0

    def resolve(self, path: str) -> str:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            return path
        try:
            resolved = path_obj.resolve(strict=False)
            rel = resolved.relative_to(self.source_root)
        except ValueError:
            return path
        cached = self.cache_root / rel
        if self._is_valid_cached_file(path_obj, cached):
            self._touch(cached)
            return str(cached)
        self._copy_with_lock(path_obj, cached)
        return str(cached)

    def prefetch(self, paths: Iterable[str]) -> None:
        if self.prefetch_window <= 0:
            return
        executor = self._get_executor()
        for path in paths:
            with self._submitted_lock:
                if path in self._submitted:
                    continue
                self._submitted.add(path)
            executor.submit(self._prefetch_one, path)

    def _prefetch_one(self, path: str) -> None:
        try:
            self.resolve(path)
        except Exception:
            # Foreground loading still raises the real error if this sample is used.
            pass
        finally:
            with self._submitted_lock:
                self._submitted.discard(path)

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.prefetch_workers,
                thread_name_prefix="dataset-cache",
            )
        return self._executor

    def _is_valid_cached_file(self, source: Path, cached: Path) -> bool:
        try:
            return cached.is_file() and cached.stat().st_size == source.stat().st_size
        except OSError:
            return False

    def _copy_with_lock(self, source: Path, cached: Path) -> None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        lock_dir = cached.with_name(cached.name + ".lock")
        while True:
            try:
                lock_dir.mkdir()
                have_lock = True
                break
            except FileExistsError:
                have_lock = False
                if self._is_valid_cached_file(source, cached):
                    self._touch(cached)
                    return
                time.sleep(0.1)
        if not have_lock:
            return
        try:
            if self._is_valid_cached_file(source, cached):
                self._touch(cached)
                return
            tmp = cached.with_name(f".{cached.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            shutil.copyfile(source, tmp)
            os.replace(tmp, cached)
            self._touch(cached)
            self._copies_since_cleanup += 1
            if self.max_cache_bytes and self._copies_since_cleanup >= 256:
                self._copies_since_cleanup = 0
                self._cleanup_if_needed()
        finally:
            try:
                lock_dir.rmdir()
            except OSError:
                pass

    def _cleanup_if_needed(self) -> None:
        if not self._cleanup_lock.acquire(blocking=False):
            return
        try:
            files: list[tuple[float, int, Path]] = []
            total = 0
            now = time.time()
            for path in self.cache_root.rglob("*"):
                if not path.is_file() or path.name.endswith(".tmp"):
                    continue
                if path.name.endswith(".lock"):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                total += stat.st_size
                last_used = max(stat.st_atime, stat.st_mtime)
                if now - last_used < _RECENT_FILE_GRACE_SECONDS:
                    continue
                files.append((stat.st_atime, stat.st_size, path))
            if total <= self.max_cache_bytes:
                return
            files.sort()
            target = int(self.max_cache_bytes * 0.9)
            for _, size, path in files:
                if total <= target:
                    break
                try:
                    path.unlink()
                    total -= size
                except OSError:
                    pass
        finally:
            self._cleanup_lock.release()

    def _touch(self, path: Path) -> None:
        try:
            os.utime(path, None)
        except OSError:
            pass
