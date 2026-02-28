import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM,AutoModelForImageTextToText

from src.data.datasets import build_dataset
from src.data.collate import collate_prompt_target_batch

DATA = r"D:\SRJ_program\program&practice\2vlmsafe\data\vlguard\processed\vlguard_train.jsonl"
MODEL = "Qwen/Qwen2-VL-2B-Instruct"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True).to(device=device,dtype=torch.float16)
    model.eval()

    ds = build_dataset(DATA, toy=False)
    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: collate_prompt_target_batch(
            b, tok, processor=proc, device=device
        ),
    )

    batch = next(iter(dl))
    print("batch keys:", batch.keys())

    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],  # Qwen2-VL 需要这个
        )

    print("loss:", float(out.loss))

if __name__ == "__main__":
    main()