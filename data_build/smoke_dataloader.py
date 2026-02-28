import json
from torch.utils.data import DataLoader
import torch
import sys
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

data_build_path = str(Path(__file__).resolve().parent)
if data_build_path not in sys.path:
    sys.path.append(data_build_path)

from src.data.datasets import build_dataset
from src.data.collate import collate_prompt_target_batch
from transformers import AutoTokenizer, AutoProcessor

DATA = r"D:\SRJ_program\program&practice\2vlmsafe\data\vlguard\processed\vlguard_train.jsonl"
MODEL = "Qwen/Qwen2-VL-2B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

ds = build_dataset(DATA, toy=False)

dl = DataLoader(
    ds,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda b: collate_prompt_target_batch(b, tok, processor=proc, device=torch.device("cpu")),
)

batch = next(iter(dl))
print("keys:", batch.keys())
print("input_ids:", batch["input_ids"].shape)
print("pixel_values:", batch["pixel_values"].shape)
print("labels:", batch["labels"].shape)
print("is_harmful:", batch["is_harmful"])