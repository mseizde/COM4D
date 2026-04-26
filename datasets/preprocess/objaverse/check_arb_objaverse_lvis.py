import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


DEFAULT_MIN_FACE_COUNT = 200
DEFAULT_MAX_FACE_COUNT = 50000
DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 5
DEFAULT_PROCESSED_DIR = Path("/data/mseizde/com4d/datasets/processed/objaverse")
BANNED_TAGS = {"character", "humanoid", "person", "animal", "creature"}
UID_FIELD_CANDIDATES = (
    "uid",
    "uids",
    "object_uid",
    "object_uids",
    "objaverse_uid",
    "objaverse_uids",
)


def _default_raw_root() -> Optional[Path]:
    candidate = Path("/data/mseizde/com4d/datasets/raw/Objaverse/hf-objaverse-v1")
    return candidate if candidate.exists() else None


def _load_json_gz(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


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


def _maybe_add_uid_string(value: str, out: List[str]) -> None:
    value = value.strip()
    if not value:
        return
    if "/" in value or "\\" in value:
        return
    if value.endswith(".glb"):
        value = Path(value).stem
    out.append(value)


def _dict_keys_look_like_uids(data: Dict[object, object]) -> bool:
    if not data:
        return False
    sample_keys = list(data.keys())[:20]
    if not all(isinstance(key, str) for key in sample_keys):
        return False
    score = 0
    for key in sample_keys:
        token = key.strip()
        if "/" in token or "\\" in token or " " in token:
            continue
        if len(token) >= 16:
            score += 1
    return score >= max(1, len(sample_keys) // 2)


def _extract_uids_from_json_obj(obj) -> List[str]:
    out: List[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        _maybe_add_uid_string(obj, out)
        return out
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                _maybe_add_uid_string(item, out)
            elif isinstance(item, dict):
                out.extend(_extract_uids_from_json_obj(item))
        return out
    if not isinstance(obj, dict):
        return out

    for field in UID_FIELD_CANDIDATES:
        if field not in obj:
            continue
        value = obj[field]
        if isinstance(value, str):
            _maybe_add_uid_string(value, out)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    _maybe_add_uid_string(item, out)
        if out:
            return out

    if _dict_keys_look_like_uids(obj):
        return [str(key).strip() for key in obj.keys()]

    for value in obj.values():
        if isinstance(value, (list, dict)):
            out.extend(_extract_uids_from_json_obj(value))
    return out


def _load_arb_uids(path: Path) -> List[str]:
    suffixes = path.suffixes
    if not path.exists():
        raise FileNotFoundError(f"Arb UID file not found: {path}")

    if path.suffix == ".txt":
        uids = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                _maybe_add_uid_string(line.split()[0], uids)
        return uids

    if path.suffix == ".jsonl":
        uids = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uids.extend(_extract_uids_from_json_obj(json.loads(line)))
        return uids

    if ".json" in suffixes:
        with path.open("r") as f:
            data = json.load(f)
        return _extract_uids_from_json_obj(data)

    raise ValueError(
        f"Unsupported Arb UID file format for {path}. Expected .txt, .json, or .jsonl."
    )


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


class LocalMetadataIndex:
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self._cache: Dict[str, Dict[str, dict]] = {}

    def get(self, uid: str, rel_path: Optional[str]) -> Optional[dict]:
        if not rel_path:
            return None
        rel_parts = Path(rel_path).parts
        if len(rel_parts) < 2:
            return None
        shard = rel_parts[1]
        if shard not in self._cache:
            shard_path = self.metadata_dir / f"{shard}.json.gz"
            if not shard_path.exists():
                self._cache[shard] = {}
            else:
                self._cache[shard] = _load_json_gz(shard_path)
        return self._cache[shard].get(uid)


def _quality_check_uid(
    uid: str,
    rel_path: Optional[str],
    raw_root: Path,
    metadata_index: LocalMetadataIndex,
    min_face_count: int,
    max_face_count: int,
    enforce_banned_tags: bool,
) -> tuple[bool, Dict[str, object]]:
    record: Dict[str, object] = {
        "uid": uid,
        "rel_path": rel_path,
        "local_path": None,
        "face_count": None,
        "fail_reason": None,
        "tags": [],
    }

    if rel_path is None:
        record["fail_reason"] = "missing_object_path"
        return False, record

    local_path = raw_root / rel_path
    record["local_path"] = str(local_path)
    if not local_path.exists():
        record["fail_reason"] = "missing_local_glb"
        return False, record

    ann = metadata_index.get(uid, rel_path)
    if ann is None:
        record["fail_reason"] = "missing_metadata"
        return False, record

    if not ann.get("isDownloadable", False):
        record["fail_reason"] = "not_downloadable"
        return False, record

    archives = ann.get("archives", {}) or {}
    glb_info = archives.get("glb", {}) or archives.get("gltf", {}) or {}
    face_count = glb_info.get("faceCount", ann.get("faceCount"))
    record["face_count"] = face_count
    if face_count is None:
        record["fail_reason"] = "missing_face_count"
        return False, record
    if face_count < min_face_count:
        record["fail_reason"] = "too_few_faces"
        return False, record
    if face_count > max_face_count:
        record["fail_reason"] = "too_many_faces"
        return False, record

    tags = sorted({t.get("name", "").lower() for t in ann.get("tags", []) if t.get("name")})
    record["tags"] = tags
    if enforce_banned_tags and set(tags) & BANNED_TAGS:
        record["fail_reason"] = "banned_tag"
        return False, record

    return True, record


def _probe_load_objects(uids: Sequence[str], raw_root: Path) -> Dict[str, Optional[str]]:
    import objaverse

    base_path = raw_root.parent
    objaverse.BASE_PATH = str(base_path)
    objaverse._VERSIONED_PATH = str(raw_root)
    objects = objaverse.load_objects(uids=list(uids), download_processes=1)
    return {str(uid): (str(path) if path is not None else None) for uid, path in objects.items()}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_categories_file = script_dir / "default_wanted_lvis.txt"
    default_report = DEFAULT_PROCESSED_DIR / "arb_objaverse_lvis_report.json"
    default_raw_root = _default_raw_root()

    parser = argparse.ArgumentParser(
        description=(
            "Read a set of Arb-Objaverse UIDs, intersect them with LVIS-labeled Objaverse UIDs, "
            "and run local non-destructive quality checks against the raw Objaverse dump."
        )
    )
    parser.add_argument(
        "--arb-uids",
        type=Path,
        required=True,
        help="Path to a .txt/.json/.jsonl file containing Arb-Objaverse UIDs.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=default_raw_root,
        help="Path to the local hf-objaverse-v1 root containing metadata and object-paths.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="LVIS categories to keep. Overrides --categories-file if provided.",
    )
    parser.add_argument(
        "--categories-file",
        type=Path,
        default=default_categories_file,
        help="Text file with one wanted LVIS category per line.",
    )
    parser.add_argument(
        "--min-face-count",
        type=int,
        default=DEFAULT_MIN_FACE_COUNT,
        help="Minimum acceptable GLB face count.",
    )
    parser.add_argument(
        "--max-face-count",
        type=int,
        default=DEFAULT_MAX_FACE_COUNT,
        help="Maximum acceptable GLB face count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for sampling printed examples and optional probes.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="How many example UIDs to print from the intersection and passing set.",
    )
    parser.add_argument(
        "--max-failure-examples",
        type=int,
        default=10,
        help="How many failing UID examples to include in the report and stdout.",
    )
    parser.add_argument(
        "--skip-banned-tags",
        action="store_true",
        help="Disable the banned-tag filter for humanoid/animal-like assets.",
    )
    parser.add_argument(
        "--probe-load-objects",
        type=int,
        default=0,
        help=(
            "Optionally call objaverse.load_objects on N passing UIDs to prove the end-to-end "
            "resolution path. This is off by default to avoid any accidental downloads."
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help=f"Optional JSON report path. If omitted, no file is written. Suggested path: {default_report}",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path for passing UID -> local GLB path.",
    )
    parser.add_argument(
        "--output-uids",
        type=Path,
        default=None,
        help="Optional text file path to write the passing UID list.",
    )
    args = parser.parse_args()

    if args.raw_root is None:
        raise ValueError("--raw-root is required when the default local Objaverse path is not present.")
    if args.categories is None:
        args.categories = _load_category_list(args.categories_file.resolve())
    return args


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    metadata_dir = raw_root / "metadata"
    lvis_path = raw_root / "lvis-annotations.json.gz"
    object_paths_path = raw_root / "object-paths.json.gz"

    if args.min_face_count < 0:
        raise ValueError("--min-face-count must be >= 0")
    if args.max_face_count < args.min_face_count:
        raise ValueError("--max-face-count must be >= --min-face-count")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0")
    if args.max_failure_examples < 0:
        raise ValueError("--max-failure-examples must be >= 0")
    if args.probe_load_objects < 0:
        raise ValueError("--probe-load-objects must be >= 0")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")
    if not lvis_path.exists():
        raise FileNotFoundError(f"LVIS annotation file not found: {lvis_path}")
    if not object_paths_path.exists():
        raise FileNotFoundError(f"Object-paths file not found: {object_paths_path}")

    rng = random.Random(args.seed)
    arb_uids = _dedupe_keep_order(_load_arb_uids(args.arb_uids.resolve()))
    print("Loaded Arb UIDs:", len(arb_uids))

    wanted_categories = {category.lower() for category in args.categories}
    print("Wanted LVIS categories:", len(wanted_categories))

    lvis = _load_json_gz(lvis_path)
    uid_to_categories: Dict[str, List[str]] = defaultdict(list)
    for category, uids in lvis.items():
        if category.lower() not in wanted_categories:
            continue
        for uid in uids:
            uid_to_categories[str(uid)].append(category)

    lvis_uid_set = set(uid_to_categories.keys())
    arb_uid_set = set(arb_uids)
    intersection_uids = sorted(arb_uid_set & lvis_uid_set)
    print("LVIS-matched Objaverse UIDs:", len(lvis_uid_set))
    print("Arb ∩ LVIS UIDs:", len(intersection_uids))

    object_paths = _load_json_gz(object_paths_path)
    metadata_index = LocalMetadataIndex(metadata_dir)

    pass_manifest: Dict[str, str] = {}
    fail_reason_counts: Counter[str] = Counter()
    failure_examples: List[Dict[str, object]] = []

    for uid in intersection_uids:
        passed, record = _quality_check_uid(
            uid=uid,
            rel_path=object_paths.get(uid),
            raw_root=raw_root,
            metadata_index=metadata_index,
            min_face_count=args.min_face_count,
            max_face_count=args.max_face_count,
            enforce_banned_tags=not args.skip_banned_tags,
        )
        record["categories"] = sorted(uid_to_categories.get(uid, []))
        if passed:
            pass_manifest[uid] = str(record["local_path"])
        else:
            reason = str(record["fail_reason"])
            fail_reason_counts[reason] += 1
            if len(failure_examples) < args.max_failure_examples:
                failure_examples.append(record)

    pass_uids = list(pass_manifest.keys())
    rng.shuffle(pass_uids)
    pass_sample = pass_uids[: args.sample_size]

    intersection_sample = list(intersection_uids)
    rng.shuffle(intersection_sample)
    intersection_sample = intersection_sample[: args.sample_size]

    intersection_category_counts: Counter[str] = Counter()
    passing_category_counts: Counter[str] = Counter()
    for uid in intersection_uids:
        intersection_category_counts.update(uid_to_categories.get(uid, []))
    for uid in pass_manifest:
        passing_category_counts.update(uid_to_categories.get(uid, []))

    probe_results = {}
    if args.probe_load_objects > 0:
        probe_uids = pass_uids[: args.probe_load_objects]
        if probe_uids:
            probe_results = _probe_load_objects(probe_uids, raw_root)
            print("load_objects probe results:")
            for uid, resolved_path in probe_results.items():
                print(f"  {uid}: {resolved_path}")
        else:
            print("load_objects probe skipped: no passing UIDs available")

    print("Passing local quality checks:", len(pass_manifest))
    if fail_reason_counts:
        print("Fail reasons:")
        for reason, count in fail_reason_counts.most_common():
            print(f"  {reason}: {count}")
    if intersection_sample:
        print("Intersection sample UIDs:", ", ".join(intersection_sample))
    if pass_sample:
        print("Passing sample UIDs:", ", ".join(pass_sample))

    report = {
        "arb_uid_count": len(arb_uids),
        "wanted_lvis_category_count": len(wanted_categories),
        "lvis_uid_count": len(lvis_uid_set),
        "intersection_uid_count": len(intersection_uids),
        "quality_pass_uid_count": len(pass_manifest),
        "quality_fail_uid_count": len(intersection_uids) - len(pass_manifest),
        "min_face_count": args.min_face_count,
        "max_face_count": args.max_face_count,
        "banned_tags_enabled": not args.skip_banned_tags,
        "intersection_sample_uids": intersection_sample,
        "passing_sample_uids": pass_sample,
        "fail_reason_counts": dict(fail_reason_counts),
        "intersection_category_counts": dict(intersection_category_counts.most_common()),
        "passing_category_counts": dict(passing_category_counts.most_common()),
        "failure_examples": failure_examples,
        "probe_load_objects_results": probe_results,
    }

    if args.output_report is not None:
        output_report = args.output_report.resolve()
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(json.dumps(report, indent=2))
        print("Wrote report:", output_report)

    if args.output_manifest is not None:
        output_manifest = args.output_manifest.resolve()
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text(json.dumps(pass_manifest, indent=2))
        print("Wrote passing manifest:", output_manifest)

    if args.output_uids is not None:
        output_uids = args.output_uids.resolve()
        output_uids.parent.mkdir(parents=True, exist_ok=True)
        output_uids.write_text("".join(f"{uid}\n" for uid in pass_manifest))
        print("Wrote passing UIDs:", output_uids)


if __name__ == "__main__":
    main()
