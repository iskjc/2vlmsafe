from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def iter_json_rows(obj: Any) -> Iterable[Dict[str, Any]]:
    # VLGuard 的 train.json 通常是 list[dict]
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    elif isinstance(obj, dict):
        # 兜底：有些数据会是 {"data":[...]}
        for k in ["data", "train", "items"]:
            v = obj.get(k)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        yield x

def safe_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return None

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def unzip_if_needed(zip_path: Path, out_dir: Path) -> None:
    # 如果 out_dir 已经有文件，就不重复解压
    if out_dir.exists() and any(out_dir.rglob("*.*")):
        return
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def resolve_image(img_root: Path, img_field: str) -> Optional[Path]:
    # img_field 可能是 "xxx.jpg" 或 "subdir/xxx.jpg"
    p = img_root / img_field
    if p.exists():
        return p
    # 兜底：只按文件名递归找（会慢一些，但数据量不大还能接受）
    name = Path(img_field).name
    hits = list(img_root.rglob(name))
    return hits[0] if hits else None

def main():
    repo_root = Path(__file__).resolve().parents[1]  # .../2vlmsafe
    vlroot = repo_root / "data" / "vlguard"
    raw_dir = ensure_dir(vlroot / "raw")
    img_dir = ensure_dir(vlroot / "images")
    out_dir = ensure_dir(vlroot / "processed")
    out_jsonl = out_dir / "vlguard_train.jsonl"

    # -------------------------
    # [1/4] 获取 train.json / train.zip：优先用 raw_dir；没有才下载（HuggingFace）
    # -------------------------
    train_json = raw_dir / "train.json"
    train_zip = raw_dir / "train.zip"

    if train_json.exists() and train_zip.exists():
        print("[1/4] use existing files in raw_dir")
        snap_path = None
    else:
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise SystemExit("缺 huggingface_hub：pip install -U huggingface_hub") from e

        print("[1/4] snapshot_download VLGuard ...")
        snap_path = Path(
            snapshot_download(
                repo_id="ys-zong/VLGuard",
                repo_type="dataset",
                local_dir=str(raw_dir),
                local_dir_use_symlinks=False,  # Windows 更稳
                allow_patterns=["train.json", "train.zip", "test.json", "test.zip"],
            )
        )
        train_json = first_existing([raw_dir / "train.json", snap_path / "train.json"]) or train_json
        train_zip = first_existing([raw_dir / "train.zip", snap_path / "train.zip"]) or train_zip

    if not train_json.exists() or not train_zip.exists():
        raise SystemExit(f"没找到 train.json/train.zip：train_json={train_json}, train_zip={train_zip}")

    print("train_json:", train_json)
    print("train_zip :", train_zip)

    # -------------------------
    # [2/4] 解压 train.zip 到 images
    #   - zip 里通常自带 train/xxx.jpg
    #   - 我们先解压到 images/，再把 images/train/* 扁平化搬到 images/*
    # -------------------------
    print("[2/4] unzip train.zip ->", img_dir)
    unzip_if_needed(train_zip, img_dir)

    # 若解压后出现 images/train/...，则把 train 下的类别目录搬到 images 根目录
    train_folder = img_dir / "train"
    if train_folder.exists() and train_folder.is_dir():
        # 如果 images 根目录下已经有 jpg/类别目录，就不动；否则搬运
        root_has_images = any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in img_dir.rglob("*.*"))
        # root_has_images 可能被 train/ 内部触发，这里再判断根目录是否已有类别文件夹
        root_has_category_dirs = any((p.is_dir() and p.name != "train") for p in img_dir.iterdir())

        if not root_has_category_dirs:
            import shutil
            for child in train_folder.iterdir():
                dst = img_dir / child.name
                if dst.exists():
                    continue
                shutil.move(str(child), str(dst))
            # 尝试删除空的 train 目录
            try:
                train_folder.rmdir()
            except OSError:
                pass

    # 选择真正的图片根：优先 images/，如果没有再退到 images/train
    img_root = img_dir
    if not any(img_root.rglob("*.jpg")) and (img_dir / "train").exists():
        img_root = img_dir / "train"

    print("image_root:", img_root)

    # -------------------------
    # [3/4] 解析 train.json -> jsonl（JSON Lines）
    # -------------------------
    print("[3/4] parse train.json -> jsonl")
    data = read_json(train_json)

    n_written = 0
    n_missing_img = 0
    n_missing_instr = 0

    with out_jsonl.open("w", encoding="utf-8") as w:
        for ex in iter_json_rows(data):
            img_field = ex.get("image")
            safe_flag = safe_bool(ex.get("safe"))
            instr_resp = ex.get("instr-resp") or ex.get("instr_resp") or ex.get("instr_resp_list")

            if not isinstance(img_field, str) or not img_field:
                n_missing_img += 1
                continue

            img_path = resolve_image(img_root, img_field)
            if img_path is None:
                n_missing_img += 1
                continue

            if safe_flag is None:
                # 没 safe 字段就跳过（你也可以改成默认 False）
                continue

            is_harmful = (not safe_flag)

            if not isinstance(instr_resp, list) or len(instr_resp) == 0:
                n_missing_instr += 1
                continue

            for item in instr_resp:
                if not isinstance(item, dict):
                    continue

                if is_harmful:
                    prompt = item.get("unsafe_instruction") or item.get("instruction")
                else:
                    prompt = item.get("safe_instruction") or item.get("instruction")

                if not isinstance(prompt, str) or not prompt.strip():
                    continue

                target = item.get("response")
                if not isinstance(target, str):
                    target = ""

                rec = {
                    "prompt": prompt.strip(),
                    "target": target,
                    "is_harmful": bool(is_harmful),
                    "image": str(img_path),
                    "harmful_category": ex.get("harmful_category", ""),
                    "harmful_subcategory": ex.get("harmful_subcategory", ""),
                    "vlguard_id": ex.get("id", ""),
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    # -------------------------
    # [4/4] 汇总输出
    # -------------------------
    print("[4/4] done")
    print("output:", out_jsonl)
    print("written:", n_written)
    print("missing_img_skipped:", n_missing_img)
    print("missing_instr_skipped:", n_missing_instr)
   

if __name__ == "__main__":
    main()