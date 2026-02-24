from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_attention(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        raise ValueError("Only .npy attention files are currently supported")

    if arr.ndim == 4:
        # [B, heads, Q, K] -> average batch and heads
        arr = arr.mean(axis=(0, 1))
    elif arr.ndim == 3:
        # [heads, Q, K] -> average heads
        arr = arr.mean(axis=0)
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported attention rank: {arr.ndim}")

    return arr


def save_heatmap(attn: np.ndarray, output_path: Path, title: str = "Attention Heatmap") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Failed to import matplotlib for visualization. "
            f"Original error: {exc}. "
            "If you recently upgraded NumPy, reinstall a compatible matplotlib build."
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Key positions")
    ax.set_ylabel("Query positions")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Save attention heatmap from a .npy tensor")
    ap.add_argument("--attention_npy", type=str, required=True)
    ap.add_argument("--output", type=str, default="outputs/attention_heatmap.png")
    ap.add_argument("--title", type=str, default="Attention Heatmap")
    args = ap.parse_args()

    path = Path(args.attention_npy)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    attn = load_attention(path)
    save_heatmap(attn, Path(args.output), title=args.title)
    print(f"saved heatmap -> {args.output}")


if __name__ == "__main__":
    main()
