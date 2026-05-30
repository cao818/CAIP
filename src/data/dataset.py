"""Dataset utilities for multimodal sarcasm detection."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor


class SarcasmDataset(Dataset):
    """Loads image-text sarcasm samples from a JSONL file."""

    def __init__(
        self,
        jsonl_path: str,
        processor: CLIPProcessor,
        max_length: int = 77,
    ) -> None:
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.root_dir = self.jsonl_path.parent
        self.processor = processor
        self.max_length = max_length
        self.samples = self._load_jsonl(self.jsonl_path)

    def _load_jsonl(self, jsonl_path: Path) -> List[Dict[str, object]]:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        samples: List[Dict[str, object]] = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_idx, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)
                missing = {"image_path", "text", "label"} - set(sample)
                if missing:
                    raise ValueError(
                        f"Missing required fields {sorted(missing)} in {jsonl_path} at line {line_idx}"
                    )
                samples.append(sample)

        if not samples:
            raise ValueError(f"No valid samples found in {jsonl_path}")
        return samples

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path).expanduser()
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()
        return path

    def __len__(self) -> int:
        return len(self.samples)

    def _load_single_sample(self, sample: Dict[str, object]) -> Dict[str, torch.Tensor]:
        image_path = self._resolve_image_path(str(sample["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        encoded = self.processor(
            text=str(sample["text"]),
            images=image,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_count = len(self.samples)

        # Try the requested index first, then roll forward to keep the dataloader alive
        # when a small number of corrupted images appear in the dataset.
        for offset in range(sample_count):
            candidate_index = (index + offset) % sample_count
            sample = self.samples[candidate_index]
            try:
                return self._load_single_sample(sample)
            except (FileNotFoundError, OSError, UnidentifiedImageError) as error:
                warnings.warn(
                    (
                        f"Skipping invalid image sample at index {candidate_index}: "
                        f"{sample.get('image_path', '<unknown>')} ({error})"
                    ),
                    stacklevel=2,
                )

        raise RuntimeError("No valid image samples could be loaded from the dataset.")


def get_dataloaders(
    train_jsonl: Optional[str] = None,
    val_jsonl: Optional[str] = None,
    test_jsonl: Optional[str] = None,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    num_workers: int = 4,
    hf_cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
) -> Dict[str, DataLoader]:
    """Builds train, validation and test dataloaders for CAIP."""

    try:
        processor = CLIPProcessor.from_pretrained(
            clip_model_name,
            cache_dir=hf_cache_dir,
            token=hf_token,
            local_files_only=local_files_only,
        )
    except OSError as error:
        raise RuntimeError(
            "Failed to load CLIPProcessor. Please either set a valid Hugging Face token, "
            "use a reachable network, or pass a local model directory via `--clip_model_name`. "
            f"Current source: {clip_model_name}"
        ) from error

    dataloaders: Dict[str, DataLoader] = {}

    if train_jsonl is not None:
        train_dataset = SarcasmDataset(train_jsonl, processor=processor)
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    if val_jsonl is not None:
        val_dataset = SarcasmDataset(val_jsonl, processor=processor)
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    if test_jsonl is not None:
        test_dataset = SarcasmDataset(test_jsonl, processor=processor)
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    if not dataloaders:
        raise ValueError("At least one of train_jsonl, val_jsonl or test_jsonl must be provided.")

    return dataloaders
