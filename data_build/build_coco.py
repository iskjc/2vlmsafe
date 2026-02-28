import json
import random
from pathlib import Path

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    repo_root = Path(__file__).resolve().parents[1]
    coco_root = repo_root / "data" / "coco"
    raw_dir = ensure_dir(coco_root / "raw")
    img_dir = ensure_dir(coco_root / "images")
    out_dir = ensure_dir(coco_root / "processed")
    out_jsonl = out_dir / "coco_train.jsonl"

    # 你需要把 COCO annotations 放到这里，例如 captions_train2017.json
    ann_path = raw_dir / "captions_train2017.json"
    if not ann_path.exists():
        raise SystemExit(f"missing annotation: {ann_path} (请把 COCO caption 标注 json 放到 data/coco/raw/)")

    # 你需要把图片放到 data/coco/images/train2017/xxxx.jpg
    train_img_root = img_dir / "train2017"
    if not train_img_root.exists():
        raise SystemExit(f"missing images dir: {train_img_root} (请把 COCO train2017 图片放到 data/coco/images/train2017/)")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    id2file = {img["id"]: img["file_name"] for img in data["images"]}

    prompts = [
        "Describe the image.",
        "Write a short caption for the image.",
        "What is happening in the image?",
    ]

    n = 0
    with out_jsonl.open("w", encoding="utf-8") as w:
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cap = ann.get("caption", "")
            if not cap or img_id not in id2file:
                continue
            file_name = id2file[img_id]
            img_path = train_img_root / file_name
            if not img_path.exists():
                continue

            rec = {
                "prompt": random.choice(prompts),
                "target": cap,
                "is_harmful": False,
                "image": str(img_path),
                "source": "coco_captions_train2017",
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print("done")
    print("output:", out_jsonl)
    print("written:", n)

if __name__ == "__main__":
    main()