from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class BuiltInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor]
    meta: Dict[str, Any]


class InputBuilder:
    """Build HF-compatible multimodal inputs as [vision][plugin][text]."""

    def __init__(self, model: nn.Module, plugin_len: int, use_position_ids: bool = True):
        self.model = model
        self.plugin_len = int(plugin_len)
        self.use_position_ids = bool(use_position_ids)

        if self.plugin_len < 0:
            raise ValueError(f"plugin_len must be >= 0, got {self.plugin_len}")

        emb_table = model.get_input_embeddings()
        if emb_table is None:
            raise ValueError("Model has no input embedding table; cannot embed text_input_ids.")
        self.text_embedding = emb_table

    def build(
        self,
        vision_embeds: torch.Tensor,
        learnable_embeds: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> BuiltInputs:
        if vision_embeds.ndim != 3:
            raise ValueError(f"vision_embeds must be rank-3 [B,V,H], got shape {tuple(vision_embeds.shape)}")
        if learnable_embeds.ndim != 3:
            raise ValueError(
                f"learnable_embeds must be rank-3 [B,N,H], got shape {tuple(learnable_embeds.shape)}"
            )
        if text_input_ids.ndim != 2:
            raise ValueError(f"text_input_ids must be rank-2 [B,T], got shape {tuple(text_input_ids.shape)}")

        device = vision_embeds.device
        dtype = vision_embeds.dtype

        batch_v, vision_len, hidden_size = vision_embeds.shape
        batch_n, plugin_len, hidden_plugin = learnable_embeds.shape
        batch_t, _ = text_input_ids.shape

        if batch_v != batch_n or batch_v != batch_t:
            raise ValueError(
                "Batch mismatch: "
                f"vision={batch_v}, plugin={batch_n}, text={batch_t}"
            )
        if hidden_plugin != hidden_size:
            raise ValueError(f"Hidden mismatch: vision H={hidden_size} vs plugin H={hidden_plugin}")
        if plugin_len != self.plugin_len:
            raise ValueError(
                f"Plugin length mismatch: builder expects {self.plugin_len}, got {plugin_len}"
            )

        model_device = self.text_embedding.weight.device
        if model_device != device:
            raise ValueError(
                "Device mismatch: model embeddings are on "
                f"{model_device}, but vision/plugin embeds are on {device}."
            )

        model_hidden = self.text_embedding.weight.shape[1]
        if model_hidden != hidden_size:
            raise ValueError(
                f"Hidden mismatch: model embedding dim={model_hidden}, vision/plugin H={hidden_size}"
            )

        text_embeds = self.text_embedding(text_input_ids.to(model_device)).to(dtype=dtype)
        text_len = text_embeds.shape[1]

        if text_attention_mask is None:
            text_attention_mask = torch.ones((batch_v, text_len), device=device, dtype=torch.long)
        else:
            if text_attention_mask.ndim != 2:
                raise ValueError(
                    f"text_attention_mask must be rank-2 [B,T], got shape {tuple(text_attention_mask.shape)}"
                )
            if text_attention_mask.shape != (batch_v, text_len):
                raise ValueError(
                    "text_attention_mask shape mismatch: "
                    f"expected {(batch_v, text_len)}, got {tuple(text_attention_mask.shape)}"
                )
            text_attention_mask = text_attention_mask.to(device=device, dtype=torch.long)

        inputs_embeds = torch.cat([vision_embeds, learnable_embeds, text_embeds], dim=1)
        full_len = vision_len + plugin_len + text_len

        vision_mask = torch.ones((batch_v, vision_len), device=device, dtype=torch.long)
        plugin_mask = torch.ones((batch_v, plugin_len), device=device, dtype=torch.long)
        attention_mask = torch.cat([vision_mask, plugin_mask, text_attention_mask], dim=1)

        position_ids = None
        attention_mask = torch.cat([vision_mask, plugin_mask, text_attention_mask], dim=1)  # [B, L]

        if self.use_position_ids:
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0).long()
        meta = {
            "B": batch_v,
            "V": vision_len,
            "N": plugin_len,
            "T": text_len,
            "L": full_len,
        }
        return BuiltInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            meta=meta,
        )
