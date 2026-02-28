#!/usr/bin/env bash
set -euo pipefail

# Simple ablation: no-op placeholder for future experiments.
# FIX(local): keep ablation script aligned with target VLM checkpoint.
python -m src.generate \
  --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
  --ckpt "outputs/learnable_tokens.pt" \
  --n_plugin 16 \
  --max_new_tokens 64
