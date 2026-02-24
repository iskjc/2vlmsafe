#!/usr/bin/env bash
set -euo pipefail

python -m src.evaluate \
  --predictions_jsonl "outputs/predictions.jsonl"
