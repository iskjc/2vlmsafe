from __future__ import annotations
from typing import Any, Dict, Sequence, List
import torch
from .datasets import PromptTargetSample


def collate_prompt_target_batch(
    batch: Sequence[PromptTargetSample],
    tokenizer: Any,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    返回可直接训练的 causal-LM batch：
      - input_ids: [B, T]
      - attention_mask: [B, T]
      - labels: [B, T]（prompt 部分为 -100）
      - is_harmful: [B]
      - prompt: add_special_tokens=True（保留 BOS 等）
      - target: add_special_tokens=False（避免第二个 BOS）
      - 先拼接，再统一 padding，保证 prompt/target 边界不被 pad 打断
    """
    if not batch:
        raise ValueError("Batch cannot be empty")

    prompts = [item.prompt for item in batch]
    targets = [item.target for item in batch]
    flags = torch.tensor([item.is_harmful for item in batch], dtype=torch.bool)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # decoder-only 常见：pad_token_id=None，用 eos 兜底
        pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id for batching.")

    input_id_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []
    prompt_lens: List[int] = []

    for p, t in zip(prompts, targets):
        p_ids = tokenizer(p, add_special_tokens=True, return_tensors="pt").input_ids[0]
        t_ids = tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids[0]

        ids = torch.cat([p_ids, t_ids], dim=0)
        labels = torch.cat([torch.full_like(p_ids, -100), t_ids], dim=0)

        input_id_list.append(ids)
        label_list.append(labels)
        attn_list.append(torch.ones_like(ids, dtype=torch.long))
        prompt_lens.append(int(p_ids.numel()))

    max_len = max(int(x.numel()) for x in input_id_list)

    def _pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
        if x.numel() == max_len:
            return x
        pad = torch.full((max_len - x.numel(),), pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([_pad_1d(x, int(pad_id)) for x in input_id_list], dim=0)
    labels = torch.stack([_pad_1d(x, -100) for x in label_list], dim=0)
    attention_mask = torch.stack([_pad_1d(x, 0) for x in attn_list], dim=0)

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "is_harmful": flags,
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        "raw_prompts": prompts,
        "raw_targets": targets,
    }

    if device is not None:
        for key, value in list(out.items()):
            if torch.is_tensor(value):
                out[key] = value.to(device)
    return out