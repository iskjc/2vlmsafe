from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VisionGateConfig:
    input_size: int
    hidden_size: int = 256
    dropout: float = 0.0
    min_scale: float = 0.0
    max_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError(f"input_size must be > 0, got {self.input_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale must be <= max_scale")


class VisionGate(nn.Module):
    """Predicts a scalar gate in [min_scale, max_scale] from vision features."""

    def __init__(self, cfg: VisionGateConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.input_size, cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 1),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        if vision_features.ndim == 3:
            pooled = vision_features.mean(dim=1)
        elif vision_features.ndim == 2:
            pooled = vision_features
        else:
            raise ValueError(
                "vision_features must be [B,V,H] or [B,H], "
                f"got shape {tuple(vision_features.shape)}"
            )

        if pooled.shape[-1] != self.cfg.input_size:
            raise ValueError(
                f"Expected feature dim {self.cfg.input_size}, got {pooled.shape[-1]}"
            )

        gate = torch.sigmoid(self.net(pooled))
        scale = self.cfg.min_scale + (self.cfg.max_scale - self.cfg.min_scale) * gate
        return scale.unsqueeze(-1)

    def apply_to_embeddings(
        self,
        learnable_embeds: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if learnable_embeds.ndim != 3:
            raise ValueError(
                "learnable_embeds must be [B,N,H], "
                f"got shape {tuple(learnable_embeds.shape)}"
            )

        scale = self.forward(vision_features)
        if scale.shape[0] != learnable_embeds.shape[0]:
            raise ValueError(
                f"Batch mismatch: gate batch={scale.shape[0]}, embeds batch={learnable_embeds.shape[0]}"
            )

        return learnable_embeds * scale, scale.squeeze(-1).squeeze(-1)
