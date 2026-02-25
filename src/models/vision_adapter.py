from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class VisionAdapterOut:
    vision_embeds: torch.Tensor  # [B, V, H]

class LlavaLikeVisionAdapter(nn.Module):
    """
    适用于“vision tower + projector -> LLM hidden size”的 VLM
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        core = getattr(model, "model", model)

        self.vision_tower = getattr(core, "vision_tower", None) or getattr(core, "vision_encoder", None)
        if self.vision_tower is None and hasattr(core, "get_vision_tower"):
            self.vision_tower = core.get_vision_tower()

        self.projector = getattr(core, "mm_projector", None) or getattr(core, "visual_projector", None)

        if self.vision_tower is None or self.projector is None:
            raise ValueError("Cannot find vision_tower/projector in model. Need a model-specific adapter.")

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, *, dtype: torch.dtype) -> VisionAdapterOut:
        """
        pixel_values: [B, C, H, W]
        return: [B, V, hidden]
        """
        # vision tower 输出: [B, V, Dv]
        vis_out = self.vision_tower(pixel_values.to(dtype=dtype))
        if hasattr(vis_out, "last_hidden_state"):
            feats = vis_out.last_hidden_state
        elif isinstance(vis_out, (tuple, list)):
            feats = vis_out[0]
        else:
            feats = vis_out

        # projector: [B, V, Dv] -> [B, V, H]
        embeds = self.projector(feats)
        return VisionAdapterOut(vision_embeds=embeds)