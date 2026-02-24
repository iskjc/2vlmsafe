#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --device "cuda" \
  --dtype "bf16" \
  --n_plugin 16 \
  --lr 5e-3 \
  --steps 200 \
  --save_path "outputs/learnable_tokens.pt"
