from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LearnableTokensConfig:
    n_tokens: int
    hidden_size: int
    init_from_token_id: Optional[int] = None
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {self.n_tokens}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if self.init_from_token_id is not None and self.init_from_token_id < 0:
            raise ValueError("init_from_token_id must be >= 0 when provided")
        if self.init_std <= 0:
            raise ValueError(f"init_std must be > 0, got {self.init_std}")


class LearnableTokens(nn.Module):
    """A trainable sequence of embeddings with shape [n_tokens, hidden_size]."""

    def __init__(self, cfg: LearnableTokensConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Parameter(torch.empty(cfg.n_tokens, cfg.hidden_size))

    @torch.no_grad()
    def initialize(self, token_embedding_table: Optional[nn.Embedding] = None) -> None:
        if self.cfg.n_tokens == 0:
            return

        if self.cfg.init_from_token_id is not None and token_embedding_table is not None:
            tid = int(self.cfg.init_from_token_id)
            vocab_size, emb_dim = token_embedding_table.weight.shape
            if tid >= vocab_size:
                raise ValueError(
                    f"init_from_token_id out of range: {tid} >= vocab_size({vocab_size})"
                )
            if emb_dim != self.cfg.hidden_size:
                raise ValueError(
                    "Embedding dim mismatch: "
                    f"token_table={emb_dim}, config.hidden_size={self.cfg.hidden_size}"
                )
            base = token_embedding_table.weight[tid]
            self.emb.copy_(base.unsqueeze(0).repeat(self.cfg.n_tokens, 1))
        else:
            self.emb.normal_(mean=0.0, std=float(self.cfg.init_std))
    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        emb = self.emb.to(device=device, dtype=dtype)
        return emb.unsqueeze(0).expand(batch_size, -1, -1)

    def save(self, path: str) -> None:
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({"emb": self.emb.detach().cpu(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str, map_location: str | torch.device = "cpu") -> None:
        try:
            ckpt = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=map_location)

        if not isinstance(ckpt, dict) or "emb" not in ckpt:
            raise ValueError(f"Invalid checkpoint format at {path}: missing 'emb'")

        emb = ckpt["emb"]
        if emb.shape != self.emb.shape:
            raise ValueError(f"Shape mismatch: ckpt {tuple(emb.shape)} vs current {tuple(self.emb.shape)}")

        with torch.no_grad():
            self.emb.copy_(emb)
