from __future__ import annotations

from typing import Any, Optional

from .base import VLMAdapter
from .generic_hf import GenericHFAdapter, GenericTemplate


class Qwen2VLAdapter(GenericHFAdapter):
    """
    Qwen2-VL needs specific conversation tokens and image placeholder tokens.
    We implement it by overriding build_texts with your proven template.
    """

    def build_texts(self, item: Any, has_image: bool):
        if has_image:
            p_text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>"
                f"{item.prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            p_text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{item.prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        full_text = p_text + f"{item.target}<|im_end|>"
        # 复用 TextPair 结构
        from .base import TextPair
        return TextPair(prompt_text=p_text, full_text=full_text)


def get_adapter(model_name: str, processor: Any, tokenizer: Any = None) -> VLMAdapter:
    name = (model_name or "").lower()

    if "qwen2-vl" in name:
        return Qwen2VLAdapter(processor=processor, tokenizer=tokenizer)

    # 其他模型：先用通用 HF adapter
    # 注意：某些模型需要 image placeholder（比如 "<image>\n"），你可在这里加规则
    # 例如 if "llava" in name: return GenericHFAdapter(..., image_placeholder="<image>\n")
    return GenericHFAdapter(
        processor=processor,
        tokenizer=tokenizer,
        template=GenericTemplate(),
        image_placeholder="",
    )