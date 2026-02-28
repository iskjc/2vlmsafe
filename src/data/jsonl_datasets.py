import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

@dataclass
class PromptTargetDataset:
    items: List[Dict[str, Any]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def load_jsonl(path: str) -> PromptTargetDataset:
    p = Path(path)
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            assert "prompt" in ex and "target" in ex and "is_harmful" in ex and "image" in ex
            items.append(ex)
    return PromptTargetDataset(items)