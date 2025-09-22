from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate answers against Benyucong/vanilla-hop2 a_entity")
    p.add_argument("--pred", required=True, help="Path to predictions JSONL (must contain id and answer)")
    p.add_argument("--hf-dataset", default="Benyucong/vanilla-hop2")
    p.add_argument("--hf-split", default="test")
    p.add_argument("--normalize", choices=["none", "simple"], default="simple")
    p.add_argument("--output", help="Optional path to write metrics JSON")
    return p.parse_args()


def normalize_text(s: str, mode: str) -> str:
    t = s.strip().strip('"').strip("'").lower()
    if mode == "simple":
        t = t.replace("_", " ")
    t = " ".join(t.split())
    return t


def to_list(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, str):
        txt = obj.strip()
        # Try JSON
        try:
            val = json.loads(txt)
            if isinstance(val, list):
                return [str(x) for x in val]
        except Exception:
            pass
        # Try Python literal
        try:
            val = ast.literal_eval(txt)
            if isinstance(val, list):
                return [str(x) for x in val]
        except Exception:
            pass
        # Fallback: single string as list
        return [txt]
    return [str(obj)]


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise SystemExit(f"Predictions not found: {pred_path}")

    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    gold_index: Dict[int, List[str]] = {}
    for row in ds:
        try:
            rid = int(row.get("id"))
        except Exception:
            continue
        gold = row.get("a_entity")
        gold_list = to_list(gold)
        gold_index[rid] = gold_list

    total = 0
    exact_match = 0
    sum_recall = 0.0
    micro_correct = 0
    micro_total_gold = 0
    missing_gold = 0
    missing_pred = 0

    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = rec.get("id")
        if rid is None:
            continue
        try:
            rid = int(rid)
        except Exception:
            continue
        gold_list = gold_index.get(rid)
        if gold_list is None:
            missing_gold += 1
            continue
        pred_list = to_list(rec.get("answer"))

        # Normalize
        gold_norm: Set[str] = set(normalize_text(x, args.normalize) for x in gold_list if str(x).strip() != "")
        pred_norm: Set[str] = set(normalize_text(x, args.normalize) for x in pred_list if str(x).strip() != "")

        if not pred_norm:
            missing_pred += 1

        total += 1
        if gold_norm == pred_norm:
            exact_match += 1

        inter = gold_norm.intersection(pred_norm)
        rec_i = (len(inter) / len(gold_norm)) if gold_norm else 1.0
        sum_recall += rec_i
        micro_correct += len(inter)
        micro_total_gold += len(gold_norm)

    accuracy = (exact_match / total) if total else 0.0
    recall_macro = (sum_recall / total) if total else 0.0
    recall_micro = (micro_correct / micro_total_gold) if micro_total_gold else 0.0

    metrics = {
        "samples": total,
        "exact_match_accuracy": accuracy,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "missing_gold_ids": missing_gold,
        "missing_pred_answers": missing_pred,
    }

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[ok] Wrote metrics to {outp}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
