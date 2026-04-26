import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import objaverse

DEFAULT_TARGET_PER_CATEGORY = 300
DEFAULT_MAX_FACE_COUNT = 50000
DEFAULT_MIN_FACE_COUNT = 200
DEFAULT_DOWNLOAD_PROCESSES = 8
DEFAULT_SEED = 42
DEFAULT_PROCESSED_DIR = Path("/data/mseizde/com4d/datasets/processed/objaverse")

BANNED_TAGS = {"character", "humanoid", "person", "animal", "creature"}


def _load_existing_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected manifest JSON object at {path}, got {type(data).__name__}")
    return {str(uid): str(local_path) for uid, local_path in data.items() if local_path is not None}


def _load_category_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Category list file not found: {path}")

    categories: List[str] = []
    with path.open("r") as f:
        for line in f:
            category = line.strip()
            if not category or category.startswith("#"):
                continue
            categories.append(category)

    if not categories:
        raise ValueError(f"No categories found in {path}")
    return categories


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output = DEFAULT_PROCESSED_DIR / "objaverse_subset_paths.json"
    default_categories_file = script_dir / "default_wanted_lvis.txt"

    parser = argparse.ArgumentParser(
        description="Filter and download a subset of Objaverse objects."
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=DEFAULT_TARGET_PER_CATEGORY,
        help="Max candidate UIDs to sample per wanted LVIS category before filtering.",
    )
    parser.add_argument(
        "--min-face-count",
        type=int,
        default=DEFAULT_MIN_FACE_COUNT,
        help="Minimum face count accepted by the filter.",
    )
    parser.add_argument(
        "--max-face-count",
        type=int,
        default=DEFAULT_MAX_FACE_COUNT,
        help="Maximum face count accepted by the filter.",
    )
    parser.add_argument(
        "--download-processes",
        type=int,
        default=DEFAULT_DOWNLOAD_PROCESSES,
        help="Number of parallel download workers passed to objaverse.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download at most this many filtered objects. Useful for a small test run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used when sampling candidate UIDs.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="LVIS categories to include. Overrides --categories-file if provided.",
    )
    parser.add_argument(
        "--categories-file",
        type=Path,
        default=default_categories_file,
        help="Path to a text file containing one LVIS category per line.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for the Objaverse cache/downloads. "
            "If set, objaverse data will be stored under <download-dir>/hf-objaverse-v1."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run filtering and manifest generation without downloading object files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Where to write the UID-to-local-path manifest JSON.",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help=(
            "If the output manifest already exists, keep its entries, exclude those UIDs "
            "from reselection, and append only newly downloaded objects."
        ),
    )
    args = parser.parse_args()
    if args.categories is None:
        args.categories = _load_category_list(args.categories_file.resolve())
    return args


def main() -> None:
    args = parse_args()
    out = args.output.resolve()

    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.target_per_category <= 0:
        raise ValueError("--target-per-category must be > 0")
    if args.min_face_count < 0:
        raise ValueError("--min-face-count must be >= 0")
    if args.max_face_count < args.min_face_count:
        raise ValueError("--max-face-count must be >= --min-face-count")
    if args.download_processes <= 0:
        raise ValueError("--download-processes must be > 0")

    existing_objects: Dict[str, str] = {}
    if args.append_existing:
        existing_objects = _load_existing_manifest(out)
        if existing_objects:
            print("Existing manifest entries:", len(existing_objects))

    if args.download_dir is not None:
        download_root = args.download_dir.resolve()
        os.makedirs(download_root, exist_ok=True)
        objaverse.BASE_PATH = str(download_root)
        objaverse._VERSIONED_PATH = str(download_root / "hf-objaverse-v1")
        print("Objaverse download root:", download_root)

    rng = random.Random(args.seed)
    wanted_lvis = {category.lower() for category in args.categories}

    # 1) Category -> [uids]
    lvis = objaverse.load_lvis_annotations()
    categories_path = out.parent / "lvis_categories.txt"
    categories_path.parent.mkdir(parents=True, exist_ok=True)
    with categories_path.open("w") as f:
        for category in sorted(lvis.keys()):
            f.write(category + "\n")
    print("Wrote LVIS categories to", categories_path)

    candidate_uids = set()
    for cat, uids in lvis.items():
        if cat.lower() in wanted_lvis:
            sampled = list(uids)
            rng.shuffle(sampled)
            candidate_uids.update(sampled[: args.target_per_category])

    if existing_objects:
        candidate_uids.difference_update(existing_objects.keys())

    candidate_uids = list(candidate_uids)
    print("Candidate UIDs before annotation filtering:", len(candidate_uids))

    # 2) Pull annotations only for candidates
    annotations = objaverse.load_annotations(candidate_uids)

    filtered = []
    for uid, ann in annotations.items():
        if not ann.get("isDownloadable", False):
            continue

        archives = ann.get("archives", {}) or {}
        glb_info = archives.get("glb", {}) or archives.get("gltf", {}) or {}

        face_count = glb_info.get("faceCount")
        if face_count is None:
            continue
        if not (args.min_face_count <= face_count <= args.max_face_count):
            continue

        # Optional tag-based cleanup
        tag_names = {t.get("name", "").lower() for t in ann.get("tags", [])}
        if tag_names & BANNED_TAGS:
            continue

        filtered.append(uid)

    print("Candidates after filtering:", len(filtered))

    if args.limit is not None:
        filtered = filtered[: args.limit]
    print("New candidates selected for download:", len(filtered))

    # 3) Download only the selected subset
    if args.dry_run:
        objects = {uid: existing_objects.get(uid) for uid in filtered}
        print("Dry run enabled: skipping downloads")
        print("Dry run would add new objects:", len(filtered))
    else:
        objects = objaverse.load_objects(
            uids=filtered,
            download_processes=args.download_processes,
        )
        print("Downloaded new objects:", len(objects))

    # 4) Save UID->path map for later preprocessing
    out.parent.mkdir(parents=True, exist_ok=True)
    combined_objects = dict(existing_objects)
    combined_objects.update(objects)
    out.write_text(json.dumps(combined_objects, indent=2))
    print("Wrote", out)
    print("Total manifest entries:", len(combined_objects))


if __name__ == "__main__":
    main()
