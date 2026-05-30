"""Data preparation script for the CAIP project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation", "dev"),
    "test": ("test", "testing"),
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def _read_nonempty_lines(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _split_key_value(line: str) -> Tuple[str, str]:
    for separator in ("\t", ",", "|"):
        if separator in line:
            key, value = line.split(separator, 1)
            return key.strip(), value.strip()
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    raise ValueError(f"Unable to parse keyed line: {line}")


def _parse_text_lines(lines: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    keyed_text: Dict[str, str] = {}
    ordered_text: List[str] = []

    for line in lines:
        try:
            key, value = _split_key_value(line)
        except ValueError:
            ordered_text.append(line)
            continue
        keyed_text[key] = value

    return keyed_text, ordered_text


def _normalize_label(label: str) -> int:
    token = label.strip().lower()
    if token in {"1", "sarcastic", "sarcasm", "yes", "true"}:
        return 1
    if token in {"0", "non-sarcastic", "nonsarcastic", "non_sarcastic", "no", "false"}:
        return 0
    try:
        return int(float(token))
    except ValueError as error:
        raise ValueError(f"Unsupported label value: {label}") from error


def _parse_label_lines(lines: Sequence[str]) -> Tuple[Dict[str, int], List[int]]:
    keyed_labels: Dict[str, int] = {}
    ordered_labels: List[int] = []

    for line in lines:
        try:
            key, value = _split_key_value(line)
        except ValueError:
            ordered_labels.append(_normalize_label(line))
            continue
        keyed_labels[key] = _normalize_label(value)

    return keyed_labels, ordered_labels


def _resolve_split_dir(raw_root: Path, split_name: str) -> Path:
    aliases = SPLIT_ALIASES[split_name]
    for alias in aliases:
        candidate = raw_root / alias
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate a directory for split '{split_name}' under {raw_root}")


def _collect_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    return image_paths


def _build_split_records(split_dir: Path) -> List[Dict[str, object]]:
    image_dir = split_dir / "image"
    text_file = split_dir / "text.txt"
    label_file = split_dir / "label.txt"

    image_paths = _collect_image_paths(image_dir)
    keyed_text, ordered_text = _parse_text_lines(_read_nonempty_lines(text_file))
    keyed_labels, ordered_labels = _parse_label_lines(_read_nonempty_lines(label_file))

    use_keyed_format = bool(keyed_text) and bool(keyed_labels)
    records: List[Dict[str, object]] = []

    if use_keyed_format:
        common_keys = [path.stem for path in image_paths if path.stem in keyed_text and path.stem in keyed_labels]
        if not common_keys:
            raise ValueError(f"No shared ids found across image/text/label files in {split_dir}")
        for key in common_keys:
            image_path = image_dir / f"{key}{next((p.suffix for p in image_paths if p.stem == key), '.jpg')}"
            records.append(
                {
                    "image_path": str(image_path),
                    "text": keyed_text[key],
                    "label": keyed_labels[key],
                }
            )
        return records

    if len(image_paths) != len(ordered_text) or len(image_paths) != len(ordered_labels):
        raise ValueError(
            f"Split {split_dir.name} has inconsistent counts: "
            f"{len(image_paths)} images, {len(ordered_text)} texts, {len(ordered_labels)} labels."
        )

    for image_path, text, label in zip(image_paths, ordered_text, ordered_labels):
        records.append(
            {
                "image_path": str(image_path),
                "text": text,
                "label": label,
            }
        )
    return records


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
    raise FileNotFoundError(f"Could not find image for image_id={image_id_str} under {image_root}")


def _build_records_from_json(json_path: Path, image_root: Path) -> List[Dict[str, object]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Split JSON not found: {json_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    with json_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a list of samples in {json_path}, got {type(raw_data).__name__}")

    records: List[Dict[str, object]] = []
    skipped_count = 0

    for idx, sample in enumerate(raw_data, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Expected dict sample at {json_path}:{idx}, got {type(sample).__name__}")

        missing = {"image_id", "text", "label"} - set(sample)
        if missing:
            raise ValueError(f"Missing required fields {sorted(missing)} in {json_path} at sample {idx}")

        try:
            image_path = _resolve_image_by_id(image_root, sample["image_id"])
        except FileNotFoundError:
            skipped_count += 1
            continue

        records.append(
            {
                "image_path": str(image_path),
                "text": str(sample["text"]).strip(),
                "label": int(sample["label"]),
            }
        )

    if not records:
        raise ValueError(f"No valid aligned records were found in {json_path}")
    if skipped_count > 0:
        print(f"[WARN] Skipped {skipped_count} samples in {json_path.name} because matching images were missing.")

    return records


def prepare_mmsd1_0(
    json_root: str,
    image_root: str,
    output_dir: str = "data/processed",
) -> Dict[str, int]:
    """Converts MMSD1.0 JSON splits and external image directory into JSONL files."""

    json_root_path = Path(json_root).expanduser().resolve()
    image_root_path = Path(image_root).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    stats: Dict[str, int] = {}

    split_to_file = {
        "train": "train.json",
        "val": "valid.json",
        "test": "test.json",
    }

    for split_name, file_name in split_to_file.items():
        records = _build_records_from_json(json_root_path / file_name, image_root_path)
        output_path = output_dir_path / f"{split_name}.jsonl"
        stats[split_name] = _write_jsonl(records, output_path)

    return stats


def prepare_mmsd2_0(raw_root: str, output_dir: str = "data/processed") -> Dict[str, int]:
    """Converts MMSD2.0 raw files into JSONL splits."""

    raw_root_path = Path(raw_root).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    stats: Dict[str, int] = {}

    for split_name in ("train", "val", "test"):
        split_dir = _resolve_split_dir(raw_root_path, split_name)
        records = _build_split_records(split_dir)
        output_path = output_dir_path / f"{split_name}.jsonl"
        stats[split_name] = _write_jsonl(records, output_path)

    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare multimodal sarcasm datasets for the CAIP project.")
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="mmsd2",
        choices=["mmsd1", "mmsd2"],
        help="Dataset layout to parse: `mmsd1` uses JSON split files, `mmsd2` supports either split directories or JSON split files.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        help="Path to the raw MMSD2.0 root directory containing train/val/test subdirectories.",
    )
    parser.add_argument(
        "--json-root",
        type=str,
        help="Path to the MMSD1.0 directory containing train.json, valid.json and test.json.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        help="Path to the shared image directory used by MMSD1.0 JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory where train.jsonl, val.jsonl and test.jsonl will be written.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.dataset_format == "mmsd1":
        if not args.json_root or not args.image_root:
            raise ValueError("MMSD1.0 preprocessing requires both --json-root and --image-root.")
        stats = prepare_mmsd1_0(
            json_root=args.json_root,
            image_root=args.image_root,
            output_dir=args.output_dir,
        )
    else:
        # MMSD2.0 is commonly released in two layouts:
        # 1) split directories: train/val/test with image/, text.txt, label.txt
        # 2) JSON split files: train.json, valid.json, test.json + a shared image directory
        #
        # To reduce friction, we auto-route based on which arguments/files exist.
        if args.json_root and args.image_root:
            stats = prepare_mmsd1_0(
                json_root=args.json_root,
                image_root=args.image_root,
                output_dir=args.output_dir,
            )
        else:
            raw_root = args.raw_root or args.json_root
            if raw_root and Path(raw_root).expanduser().resolve().joinpath("train.json").exists():
                if not args.image_root:
                    raise ValueError(
                        "Detected MMSD2.0 JSON split files (train.json/valid.json/test.json) but missing --image-root."
                    )
                stats = prepare_mmsd1_0(
                    json_root=raw_root,
                    image_root=args.image_root,
                    output_dir=args.output_dir,
                )
            else:
                if not raw_root:
                    raise ValueError(
                        "MMSD2.0 preprocessing requires --raw-root (split directories) or (--json-root and --image-root) (JSON splits)."
                    )
                stats = prepare_mmsd2_0(raw_root=raw_root, output_dir=args.output_dir)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
