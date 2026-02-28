#!/usr/bin/env bash
set -euo pipefail

# FIX(local): default attention file aligns with extract_attention output.
python -m src.models.visualize_attention \
  --attention_npy "outputs/attn_demo/plug_attn.npy" \
  --output "outputs/attention_heatmap.png"
