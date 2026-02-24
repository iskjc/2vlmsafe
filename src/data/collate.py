from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from .datasets import PromptTargetSample


def collate_prompt_target_batch(
    batch: Sequence[PromptTargetSample],
    tokenizer: Any,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Batch cannot be empty")

    prompts = [item.prompt for item in batch]
    targets = [item.target for item in batch]
    flags = torch.tensor([item.is_harmful for item in batch], dtype=torch.bool)

    prompt_enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    target_enc = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)

    out: Dict[str, Any] = {
        "prompt_input_ids": prompt_enc.input_ids,
        "prompt_attention_mask": prompt_enc.attention_mask,
        "target_input_ids": target_enc.input_ids,
        "target_attention_mask": target_enc.attention_mask,
        "is_harmful": flags,
        "raw_prompts": prompts,
        "raw_targets": targets,
    }

    if device is not None:
        for key, value in list(out.items()):
            if torch.is_tensor(value):
                out[key] = value.to(device)

    return out
