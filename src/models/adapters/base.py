from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class TextPair:
    prompt_text: str
    full_text: str


class VLMAdapter:
    """
    Adapter interface for different Vision-Language Models (VLMs).

    Collate only needs:
      - build_texts(item, has_image) -> TextPair
      - encode(texts, images) -> batch dict
      - encode_prompt(texts, images) -> batch dict (for prompt length masking)
    """

    def __init__(self, processor: Any, tokenizer: Any = None):
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)

    def build_texts(self, item: Any, has_image: bool) -> TextPair:
        raise NotImplementedError

    def encode(self, texts: Sequence[str], images: Optional[Sequence[Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def encode_prompt(self, prompt_texts: Sequence[str], images: Optional[Sequence[Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def supports_text_only(processor: Any, tokenizer: Any) -> bool:
        return tokenizer is not None

    @staticmethod
    def safe_get(item: Any, name: str, default=None):
        return getattr(item, name, default)