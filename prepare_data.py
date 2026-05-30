"""Convert MMSD1.0/MMSD2.0 JSON splits into JSONL files used by CAIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def _write_jsonl(records: Iterable[Dict[str, object]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _resolve_image_by_id(image_root: Path, image_id: object) -> Path:
    image_id_str = str(image_id)
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / f"{image_id_str}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for image_id={image_id_str} under {image_root}")


def _load_split_json(json_root: Path, split_name: str) -> list[dict]:
    candidates = {
        "train": ("train.json",),
        "val": ("valid.json", "val.json"),
        "test": ("test.json",),
    }[split_name]
    for name in candidates:
        path = json_root / name
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
            return data
    raise FileNotFoundError(f"Missing {split_name} split JSON under {json_root}")


def convert_mmsd_json_splits(
    json_root: Path,
    image_root: Path,
    output_dir: Path,
    *,
    skip_missing_images: bool,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        raw_samples = _load_split_json(json_root, split_name)
        records: list[Dict[str, object]] = []
        skipped = 0
        for sample in raw_samples:
            if not isinstance(sample, dict):
                raise ValueError(f"Expected dict sample in {json_root} {split_name}, got {type(sample).__name__}")
            missing = {"image_id", "text", "label"} - set(sample)
            if missing:
                raise ValueError(f"Missing fields {sorted(missing)} in {json_root} {split_name}")
            try:
                image_path = _resolve_image_by_id(image_root, sample["image_id"])
            except FileNotFoundError:
                if not skip_missing_images:
                    raise
                skipped += 1
                continue
            records.append(
                {
                    "image_path": str(image_path),
                    "text": str(sample["text"]).strip(),
                    "label": int(sample["label"]),
                }
            )
        if not records:
            raise ValueError(f"No records produced for {json_root} {split_name}")
        if skipped:
            print(f"[WARN] {json_root.name} {split_name}: skipped {skipped} samples due to missing images.")
        stats[split_name] = _write_jsonl(records, output_dir / f"{split_name}.jsonl")
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert MMSD1.0/MMSD2.0 JSON splits to CAIP JSONL.")
    parser.add_argument("--mmsd1-root", type=str, help="Path to MMSD1.0 directory containing train/valid/test.json.")
    parser.add_argument("--mmsd2-root", type=str, help="Path to MMSD2.0 directory containing train/valid/test.json.")
    parser.add_argument("--image-root", type=str, help="Path to dataset_image directory containing images.")
    parser.add_argument("--output-root", type=str, help="Output root directory. Writes into <output-root>/MMSD1.0 and MMSD2.0.")
    parser.add_argument("--skip-missing-images", action="store_true", help="Skip samples whose images are missing.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parent
    work_root = repo_root.parent

    mmsd1_root = Path(args.mmsd1_root).expanduser().resolve() if args.mmsd1_root else work_root / "MMSD1.0"
    mmsd2_root = Path(args.mmsd2_root).expanduser().resolve() if args.mmsd2_root else work_root / "MMSD2.0"
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else work_root / "dataset_image"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else repo_root / "data"

    if not image_root.exists():
        raise FileNotFoundError(f"image-root not found: {image_root}")

    stats = {
        "MMSD1.0": convert_mmsd_json_splits(
            json_root=mmsd1_root,
            image_root=image_root,
            output_dir=output_root / "MMSD1.0",
            skip_missing_images=args.skip_missing_images,
        ),
        "MMSD2.0": convert_mmsd_json_splits(
            json_root=mmsd2_root,
            image_root=image_root,
            output_dir=output_root / "MMSD2.0",
            skip_missing_images=args.skip_missing_images,
        ),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
