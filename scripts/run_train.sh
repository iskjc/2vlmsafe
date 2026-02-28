#!/usr/bin/env bash
set -euo pipefail

# FIX(local): use target VLM checkpoint by default.
python -m src.train \
  --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
  --device "cuda" \
  --dtype "bf16" \
  --n_plugin 16 \
  --lr 5e-3 \
  --steps 200 \
  --save_path "outputs/learnable_tokens.pt"
