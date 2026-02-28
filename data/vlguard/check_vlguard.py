# check_vlguard.py
# 验证 VLGuard 数据集（JSONL: JavaScript Object Notation Lines）
# 1) JSONL 每行可解析
# 2) 必备字段存在 & 类型正确
# 3) harmful/harmless 分布统计
# 4) 自动探测图片字段并检查图片是否在 images/ 下存在
# 5) 随机抽样打印，人工 sanity check（合理性检查）

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ANN_TRAIN = BASE_DIR / "processed" / "vlguard_train_90.jsonl"
ANN_VAL   = BASE_DIR / "processed" / "vlguard_val_10.jsonl"
IMG_ROOT  = BASE_DIR / "images"



REQUIRED_KEYS = ("prompt", "target", "is_harmful")  # 你之前的格式
IMAGE_KEY_CANDIDATES = (
    "image", "image_path", "img", "img_path", "filepath", "file_name", "filename", "path", "image_id"
)

SAMPLE_PRINT_N = 5          # 打印几条样本看内容
RANDOM_CHECK_N = 200        # 随机检查多少条图片是否存在


def load_jsonl(path: Path) -> list[dict]:
    assert path.exists(), f"找不到标注文件: {path.resolve()}"
    items = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"[BAD JSON] {path.name} line {i}: {e}")
    if bad_lines:
        print(f"[WARN] {path.name} 有 {bad_lines} 行不是合法 JSON（JavaScript Object Notation）")
    return items


def detect_image_key(example: dict) -> str | None:
    for k in IMAGE_KEY_CANDIDATES:
        if k in example:
            return k
    return None


def build_image_name_set(img_root: Path) -> set[str]:
    assert img_root.exists(), f"找不到 images 目录: {img_root.resolve()}"
    names = set()
    for p in img_root.rglob("*"):
        if p.is_file():
            names.add(p.name)  # 只用文件名匹配，避免路径写法不一致导致误判
    return names


def validate_items(items: list[dict], split_name: str):
    total = len(items)
    missing_key_count = 0
    type_error_count = 0
    harmful = 0
    harmless = 0

    for idx, o in enumerate(items, 1):
        # 必备字段
        if not all(k in o for k in REQUIRED_KEYS):
            missing_key_count += 1
            continue

        # is_harmful 必须是 bool
        if not isinstance(o["is_harmful"], bool):
            type_error_count += 1
            continue

        if o["is_harmful"]:
            harmful += 1
        else:
            harmless += 1

    print(f"\n=== Split: {split_name} ===")
    print("total:", total)
    print("harmful:", harmful)
    print("harmless:", harmless)
    print("missing_required_keys:", missing_key_count)
    print("type_errors(is_harmful not bool):", type_error_count)

    if total > 0:
        print("harmful_ratio:", harmful / total)

    # 随机打印几条，人工检查语义是否合理
    print(f"\n[Sample {SAMPLE_PRINT_N}]")
    for o in random.sample(items, k=min(SAMPLE_PRINT_N, total)):
        # 只打印关键字段，避免太长
        show = {k: o.get(k) for k in REQUIRED_KEYS}
        # 可能存在图片字段也顺带打印
        img_k = detect_image_key(o)
        if img_k:
            show[img_k] = o.get(img_k)
        print(json.dumps(show, ensure_ascii=False))


def check_image_alignment(items: list[dict], split_name: str, img_root: Path):
    if not items:
        print(f"\n[SKIP] {split_name}: 没有样本，跳过图片对齐检查")
        return

    img_key = detect_image_key(items[0])
    if img_key is None:
        print(f"\n[SKIP] {split_name}: 没检测到图片字段（候选={IMAGE_KEY_CANDIDATES}）")
        print("你可以打开一条样本看看它的 key 是啥，然后把 IMAGE_KEY_CANDIDATES 加进去。")
        print("example keys:", sorted(items[0].keys()))
        return

    all_img_names = build_image_name_set(img_root)

    checked = 0
    missing = 0
    missing_examples = []

    sample_items = random.sample(items, k=min(RANDOM_CHECK_N, len(items)))
    for o in sample_items:
        v = o.get(img_key, "")
        name = Path(str(v)).name  # 不管写的是绝对路径/相对路径，只取最后的文件名
        checked += 1
        if name not in all_img_names:
            missing += 1
            if len(missing_examples) < 10:
                missing_examples.append((v, name))

    print(f"\n=== Image Alignment: {split_name} ===")
    print("img_root:", img_root.resolve())
    print("detected_img_key:", img_key)
    print("checked:", checked)
    print("missing:", missing)
    print("missing_rate:", (missing / checked) if checked else 0.0)

    if missing_examples:
        print("\n[Missing examples: show up to 10]")
        for raw_v, just_name in missing_examples:
            print("  raw_field_value:", raw_v, " | parsed_name:", just_name)


def main():
    print("Working dir:", Path.cwd())
    print("Train ann:", ANN_TRAIN.resolve())
    print("Val ann:", ANN_VAL.resolve())
    print("Images root:", IMG_ROOT.resolve())

    train_items = load_jsonl(ANN_TRAIN)
    val_items = load_jsonl(ANN_VAL)

    validate_items(train_items, "train_90")
    validate_items(val_items, "val_10")

    check_image_alignment(train_items, "train_90", IMG_ROOT)
    check_image_alignment(val_items, "val_10", IMG_ROOT)

    print("\nDone. 如果 missing_rate > 0，基本就是标注里的图片名/后缀/路径和 images/ 不一致。")


if __name__ == "__main__":
    main()