#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, sys
from pathlib import Path
from typing import Dict, Any, List
from datasets import Dataset

PAT_TRAILING_IMAGE = re.compile(r"\s*<image>\s*$", re.IGNORECASE)

def load_json_array(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} root must be a JSON array")
    return data

def pick_first_human(convs: List[Dict[str, Any]]) -> str:
    for m in convs:
        if m.get("from") == "human":
            return m.get("value", "")
    raise ValueError("no human message found")

def pick_last_gpt(convs: List[Dict[str, Any]]) -> str:
    for m in reversed(convs):
        if m.get("from") == "gpt":
            return m.get("value", "")
    raise ValueError("no gpt message found")

def clean_prompt(text: str) -> str:

    return PAT_TRAILING_IMAGE.sub("", text).rstrip()

def convert_item(obj: Dict[str, Any], data_source: str) -> Dict[str, Any]:
    img = obj.get("image")
    if img is None:
        raise ValueError("missing 'image'")
    # images = img if isinstance(img, list) else [img]
    raw_images = img if isinstance(img, list) else [img]
    images = [{"image": p} for p in raw_images]

    convs = obj.get("conversations") or []
    # human = clean_prompt(pick_first_human(convs))
    human = pick_first_human(convs)
    gpt   = pick_last_gpt(convs)

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": human}],
        "images": images,
        "reward_model": {"style": "rule", "ground_truth": gpt},
    }

def convert_file(in_path: str, out_path: str, data_source: str) -> int:
    raw = load_json_array(in_path)
    out = []
    bad = 0
    for i, obj in enumerate(raw):
        try:
            out.append(convert_item(obj, data_source))
        except Exception as e:
            bad += 1
            print(f"[WARN] skip idx={i}: {e}", file=sys.stderr)
    Dataset.from_list(out).to_parquet(out_path)
    print(f"[OK] {in_path} -> {out_path} (ok={len(out)}, skipped={bad})")
    return len(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_source", default="custom/vla_r1")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_train = convert_file(args.train_json, os.path.join(args.output_dir, "train.parquet"), args.data_source)
    n_val   = convert_file(args.val_json,   os.path.join(args.output_dir, "test.parquet"),  args.data_source)
    print(f"Done. train={n_train}, test={n_val}")

if __name__ == "__main__":
    main()


"""
python share_robot_to_parquet.py \
  --train_json path/to/train.json \
  --val_json   path/to/val.json \
  --output_dir path/to/output_dir \
  --data_source custom/vla_r1
"""