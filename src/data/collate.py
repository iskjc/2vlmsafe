from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

try:
    from src.models.adapters.registry import get_adapter
except ModuleNotFoundError:
    # FIX(local): allow direct execution paths where package prefix `src.` is unavailable.
    from models.adapters.registry import get_adapter


def collate_unified(
    batch: Sequence[Any],
    processor: Any,
    model_name: str,
    device: Optional[torch.device] = None,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    adapter = get_adapter(model_name=model_name, processor=processor, tokenizer=tokenizer)

    images: List[Optional[Image.Image]] = []
    prompt_texts: List[str] = []
    full_texts: List[str] = []
    has_flags: List[bool] = []

    for item in batch:
        img_val = getattr(item, "image_path", None) or getattr(item, "image", None)
        has_img = (img_val is not None)
        has_flags.append(has_img)

        if has_img:
            img_path = Path(img_val)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at: {img_path}")
            images.append(Image.open(img_path).convert("RGB"))
        else:
            images.append(None)

        pair = adapter.build_texts(item=item, has_image=has_img)
        prompt_texts.append(pair.prompt_text)
        full_texts.append(pair.full_text)

    all_have = all(has_flags)
    none_have = not any(has_flags)
    if not (all_have or none_have):
        raise ValueError(
            "Mixed batch: some samples have images and some do not. "
            "Keep modality-consistent batches."
        )

    # encode 两次：full + prompt
    if all_have:
        batch_images = [im for im in images if im is not None]
        batch_data = adapter.encode(texts=full_texts, images=batch_images)
        prompt_data = adapter.encode_prompt(prompt_texts=prompt_texts, images=batch_images)
    else:
        batch_data = adapter.encode(texts=full_texts, images=None)
        prompt_data = adapter.encode_prompt(prompt_texts=prompt_texts, images=None)

    input_ids = batch_data["input_ids"]
    attention_mask = batch_data["attention_mask"]

    labels = input_ids.clone()
    for i in range(len(batch)):
        p_len = int(prompt_data["attention_mask"][i].sum().item())
        labels[i, :p_len] = -100

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "is_harmful": torch.tensor([bool(getattr(item, "is_harmful")) for item in batch], dtype=torch.bool),
    }

    if all_have:
        if "pixel_values" in batch_data:
            out["pixel_values"] = batch_data["pixel_values"]
        if "image_grid_thw" in batch_data:
            out["image_grid_thw"] = batch_data["image_grid_thw"]

    if device is not None:
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}

    return out


def collate_prompt_target_batch(
    batch: Sequence[Any],
    tokenizer: Any = None,
    processor: Any = None,
    model_name: str = "",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    # FIX(local): keep backward-compatible API used by train/data_build scripts.
    # Some callers pass tokenizer positionally, others by keyword.
    if processor is None:
        raise ValueError("processor must not be None")
    if not model_name:
        # FIX(local): infer model name for legacy callers that do not pass model_name.
        model_name = (
            getattr(processor, "name_or_path", "")
            or getattr(tokenizer, "name_or_path", "")
            or ""
        )
    return collate_unified(
        batch=batch,
        processor=processor,
        model_name=model_name,
        device=device,
        tokenizer=tokenizer,
    )
