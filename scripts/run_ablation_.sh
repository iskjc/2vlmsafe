#!/usr/bin/env bash
set -euo pipefail

# Simple ablation: no-op placeholder for future experiments.
python -m src.generate \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --ckpt "outputs/learnable_tokens.pt" \
  --n_plugin 16 \
  --max_new_tokens 64
