from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Join local 3-hop JSONL with HF dataset by id and split predictions.")
    ap.add_argument("--input", required=True, help="Path to local 3-hop JSONL (must contain id and prediction).")
    ap.add_argument("--output", required=True, help="Path to write joined JSONL with path1..path3.")
    ap.add_argument("--hf-dataset", default="Benyucong/vanilla-hop3", help="HF dataset repo id for 3-hop.")
    ap.add_argument("--hf-split", default="test", help="HF dataset split to use (e.g., train/validation/test).")
    return ap.parse_args()


def build_hf_index(ds_repo: str, split: str) -> Dict[int, Dict[str, Any]]:
    """Build an index of HF dataset rows by id."""
    try:
        ds = load_dataset(ds_repo, split=split)
        index: Dict[int, Dict[str, Any]] = {}
        for row in ds:
            # Expect an integer id
            rid = int(row.get("id"))
            index[rid] = row
        return index
    except Exception as e:
        print(f"[warn] Failed to load HF dataset {ds_repo} split={split}: {e}")
        print("[warn] Will proceed with local data only")
        return {}


def normalize_paths_3hop(prediction: Any) -> List[str]:
    """
    Parse prediction field for 3-hop paths.
    Expected format: prediction is a list of path arrays, e.g., [[p1,p2,p3], [p1,p2,p3], ...]
    Each path array should have 3 elements.
    Returns up to 3 paths as comma-separated strings.
    """
    paths: List[str] = []
    if isinstance(prediction, list):
        for item in prediction[:3]:  # Take first 3 paths
            if isinstance(item, list):
                # Ensure we have exactly 3 relations for 3-hop
                if len(item) >= 3:
                    paths.append(", ".join(str(x) for x in item[:3]))
                elif len(item) > 0:
                    # If less than 3 but not empty, still include it
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

    processed = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            rid = int(rec.get("id"))
            hf_row = hf_index.get(rid, {})

            # Extract up to 3 paths from local prediction
            paths = normalize_paths_3hop(rec.get("prediction", []))
            
            # Pad to 3 paths if needed
            while len(paths) < 3:
                paths.append("")

            # Start from all HF columns; if missing, fall back to at least id and question
            if hf_row:
                out = dict(hf_row)
            else:
                out = {
                    "id": rid,
                    "question": rec.get("question", ""),
                }
            
            # Append flattened paths
            out["path1"] = paths[0]
            out["path2"] = paths[1]
            out["path3"] = paths[2]
            
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            processed += 1

    print(f"[ok] Processed {processed} rows, skipped {skipped}")
    print(f"[ok] Output written to {out_path}")


if __name__ == "__main__":
    main()
