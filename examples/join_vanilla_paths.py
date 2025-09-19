from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Join local JSONL with HF dataset by id and split predictions.")
    ap.add_argument("--input", required=True, help="Path to local JSONL (must contain id and prediction).")
    ap.add_argument("--output", required=True, help="Path to write joined JSONL with path1..path3.")
    ap.add_argument("--hf-dataset", default="Benyucong/vanilla-hop2", help="HF dataset repo id.")
    ap.add_argument("--hf-split", default="train", help="HF dataset split to use (e.g., train/validation/test).")
    return ap.parse_args()


def build_hf_index(ds_repo: str, split: str) -> Dict[int, Dict[str, Any]]:
    ds = load_dataset(ds_repo, split=split)
    index: Dict[int, Dict[str, Any]] = {}
    for row in ds:
        # Expect an integer id
        rid = int(row.get("id"))
        index[rid] = row
    return index


def normalize_paths(prediction: Any) -> List[str]:
    # prediction is expected to be a list of path arrays, e.g., [[p1,p2], [p1], ...]
    paths: List[str] = []
    if isinstance(prediction, list):
        for item in prediction[:3]:
            if isinstance(item, list):
                paths.append(", ".join(str(x) for x in item))
            else:
                paths.append(str(item))
    return paths


def main() -> None:
    args = parse_args()

    hf_index = build_hf_index(args.hf_dataset, args.hf_split)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = int(rec.get("id"))
            hf_row = hf_index.get(rid, {})

            # Extract up to 3 paths from local prediction
            paths = normalize_paths(rec.get("prediction", []))
            while len(paths) < 3:
                paths.append("")

            # Start from all HF columns; if missing, fall back to at least id
            out = dict(hf_row) if hf_row else {"id": rid}
            # Append flattened paths
            out["path1"] = paths[0]
            out["path2"] = paths[1]
            out["path3"] = paths[2]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
