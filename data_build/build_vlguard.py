# data_build/build_vlguard.py
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

    # 1) 下载 train.json / train.zip 到 raw_dir
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

    train_json = first_existing([raw_dir / "train.json", snap_path / "train.json"])
    train_zip  = first_existing([raw_dir / "train.zip",  snap_path / "train.zip"])

    if train_json is None or train_zip is None:
        raise SystemExit(f"没找到 train.json/train.zip。raw_dir={raw_dir}, snap_path={snap_path}")

    print("train_json:", train_json)
    print("train_zip :", train_zip)

    # 2) 解压图片到 img_dir/train
    train_img_root = img_dir / "train"
    print("[2/4] unzip train.zip ->", train_img_root)
    unzip_if_needed(train_zip, train_img_root)

    # 3) 解析 train.json，生成统一格式 jsonl
    print("[3/4] parse train.json -> jsonl")
    data = read_json(train_json)

    # VLGuard 里你日志显示有字段：
    # id, image, safe(bool), harmful_category, harmful_subcategory, instr-resp(list of structs)
    # 其中 instr-resp 每个 item 可能有 instruction/response/safe_instruction/unsafe_instruction
    # 我们策略：
    # - 每个 instr-resp item 生成一条样本
    # - is_harmful = not safe
    # - prompt 优先用 unsafe_instruction / safe_instruction；退化用 instruction
    # - target 优先用 response；没有就空字符串
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

            img_path = resolve_image(train_img_root, img_field)
            if img_path is None:
                n_missing_img += 1
                continue

            if safe_flag is None:
                # 没 safe 字段就跳过（你也可以设默认 False）
                continue

            is_harmful = (not safe_flag)

            if not isinstance(instr_resp, list) or len(instr_resp) == 0:
                n_missing_instr += 1
                continue

            for item in instr_resp:
                if not isinstance(item, dict):
                    continue

                prompt = None
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
                    # 额外信息可留着以后分析
                    "harmful_category": ex.get("harmful_category", ""),
                    "harmful_subcategory": ex.get("harmful_subcategory", ""),
                    "vlguard_id": ex.get("id", ""),
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print("[4/4] done")
    print("output:", out_jsonl)
    print("written:", n_written)
    print("missing_img_skipped:", n_missing_img)
    print("missing_instr_skipped:", n_missing_instr)

if __name__ == "__main__":
    main()