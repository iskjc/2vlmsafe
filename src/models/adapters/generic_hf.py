from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, List

from .base import VLMAdapter, TextPair


@dataclass
class GenericTemplate:
    # 当 tokenizer 没有 chat_template 时使用
    system: str = "You are a helpful assistant."
    user_prefix: str = "User: "
    assistant_prefix: str = "Assistant: "
    end: str = ""  # 你也可以设成 "\n"


class GenericHFAdapter(VLMAdapter):
    """
    Generic adapter for HuggingFace-style (HF, HuggingFace) processors/tokenizers.

    Strategy:
      - If tokenizer.apply_chat_template exists: use chat template for robustness
      - Else: use a simple text template
      - Images:
          - If has_image: call processor(text=..., images=...)
          - Else: tokenizer(text=...) only
    """

    def __init__(
        self,
        processor: Any,
        tokenizer: Any = None,
        template: Optional[GenericTemplate] = None,
        # When has_image=True, some models need an explicit image placeholder in text.
        # We keep it configurable; default is empty (no placeholder).
        image_placeholder: str = "",
    ):
        super().__init__(processor=processor, tokenizer=tokenizer)
        self.template = template or GenericTemplate()
        self.image_placeholder = image_placeholder

    def _build_with_chat_template(self, prompt: str, target: str, has_image: bool) -> TextPair:
        # 这里不假设具体视觉 token，只提供可配置 placeholder
        user_content = (self.image_placeholder + prompt) if has_image else prompt

        messages = [
            {"role": "system", "content": self.template.system},
            {"role": "user", "content": user_content},
        ]

        # prompt_text：到 assistant 开始为止（用 add_generation_prompt=True）
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # full_text：prompt_text + target（让 labels 只在 target 计算）
        full_text = prompt_text + str(target)
        return TextPair(prompt_text=prompt_text, full_text=full_text)

    def _build_simple(self, prompt: str, target: str, has_image: bool) -> TextPair:
        user = (self.image_placeholder + prompt) if has_image else prompt
        p = (
            f"{self.template.system}\n"
            f"{self.template.user_prefix}{user}\n"
            f"{self.template.assistant_prefix}"
        )
        full = p + str(target) + self.template.end
        return TextPair(prompt_text=p, full_text=full)

    def build_texts(self, item: Any, has_image: bool) -> TextPair:
        prompt = getattr(item, "prompt")
        target = getattr(item, "target")

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self._build_with_chat_template(prompt, target, has_image)
        return self._build_simple(prompt, target, has_image)

    def encode(self, texts: Sequence[str], images: Optional[Sequence[Any]]) -> Dict[str, Any]:
        if images is not None:
            return self.processor(text=list(texts), images=list(images), return_tensors="pt", padding=True)
        # text-only
        tok = self.tokenizer
        if tok is None:
            raise ValueError("Text-only batch but tokenizer is None.")
        return tok(list(texts), return_tensors="pt", padding=True, truncation=True)

    def encode_prompt(self, prompt_texts: Sequence[str], images: Optional[Sequence[Any]]) -> Dict[str, Any]:
        # 与 encode 同步，保证 attention_mask 的长度统计一致
        return self.encode(prompt_texts, images)