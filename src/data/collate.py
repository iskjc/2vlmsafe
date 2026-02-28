from __future__ import annotations

from typing import Any, Dict, Sequence, List
from pathlib import Path

import torch
from .datasets import PromptTargetSample

def collate_prompt_target_batch(
    batch: Sequence[PromptTargetSample],
    tokenizer: Any,
    processor: Any | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Batch cannot be empty")
    if processor is None:
        raise ValueError("Qwen2-VL requires a processor to handle dynamic image tokens.")

    from PIL import Image
    
    # 1. 准备图片和构建对话文本
    images = []
    full_texts = []
    prompt_texts = []

    for item in batch:
        img_val = getattr(item, "image_path", None) or getattr(item, "image", None)

        if img_val is None:
            raise AttributeError(f"Sample object has no image_path or image attribute: {item}")
        img_path = Path(img_val)

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path}")
        images.append(Image.open(img_path).convert("RGB"))

        # 构造完整对话 (用于生成 input_ids)
        # 注意：Qwen2-VL 的模板非常关键
        p_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item.prompt}<|im_end|>\n<|im_start|>assistant\n"
        t_text = f"{item.target}<|im_end|>"
        full_texts.append(p_text + t_text)
        prompt_texts.append(p_text)

    # 2. 调用 Processor 处理所有数据
    batch_data = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # 3. 计算 Labels (掩盖 Prompt 部分的 Loss)
    # 再把只含 Prompt 的文本处理一遍，以此确定每个 sample 中 Prompt 占据的 token 长度
    prompt_data = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    input_ids = batch_data["input_ids"]
    labels = input_ids.clone()
    
    # 将 Prompt 对应的部分全部设为 -100
    for i in range(len(batch)):
        # 找到 prompt 的长度（注意：processor 返回的是补齐后的，这里需要通过 attention_mask 算实际长度）
        p_len = prompt_data["attention_mask"][i].sum().item()
        labels[i, :p_len] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": batch_data["attention_mask"],
        "labels": labels,
        "pixel_values": batch_data["pixel_values"],
        "image_grid_thw": batch_data.get("image_grid_thw"), # Qwen2-VL 特有
        "is_harmful": torch.tensor([item.is_harmful for item in batch], dtype=torch.bool),
    }

    if device is not None:
        out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}

    return out