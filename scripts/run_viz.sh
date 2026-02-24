#!/usr/bin/env bash
set -euo pipefail

python -m src.visualize_attention \
  --attention_npy "outputs/attention.npy" \
  --output "outputs/attention_heatmap.png"
