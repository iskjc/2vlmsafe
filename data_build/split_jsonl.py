from pathlib import Path
import json, random

src = Path(r"D:\SRJ_program\program&practice\2vlmsafe\data\vlguard\processed\vlguard_train.jsonl")
out_train = src.parent / "vlguard_train_90.jsonl"
out_val   = src.parent / "vlguard_val_10.jsonl"

random.seed(42)
rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
random.shuffle(rows)

n = len(rows)
k = int(n * 0.9)

def dump(path, items):
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

dump(out_train, rows[:k])
dump(out_val, rows[k:])

print("total:", n, "train:", k, "val:", n-k)
print("train:", out_train)
print("val  :", out_val)