from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Set

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a single naive 3-hop prediction file (answers_merged_naive_seq_3hop.jsonl) against Benyucong/vanilla-hop3."
    )
    p.add_argument(
        "--pred", default="outputs/answers_merged_naive_seq_3hop.jsonl", help="Predictions JSONL path (default outputs/answers_merged_naive_seq_3hop.jsonl)"
    )
    p.add_argument("--hf-dataset", default="Benyucong/vanilla-hop3")
    p.add_argument("--hf-split", default="test")
    p.add_argument("--normalize", choices=["none", "simple"], default="none")
    p.add_argument(
        "--output-json", help="Optional JSON file to write metrics (aggregated)."
    )
    p.add_argument(
        "--output-csv", help="Optional CSV file to write a single-row metrics record."
    )
    return p.parse_args()


def normalize_text(s: str, mode: str) -> str:
    t = s.strip().strip('"').strip("'").lower()
    if mode == "simple":
        t = t.replace("_", " ")
    return " ".join(t.split())


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
        return [txt]
    return [str(obj)]


def load_gold(dataset: str, split: str) -> Dict[int, List[str]]:
    ds = load_dataset(dataset, split=split)
    gold: Dict[int, List[str]] = {}
    for row in ds:
        try:
            rid = int(row.get("id"))
        except Exception:
            continue
        gold[rid] = to_list(row.get("a_entity"))
    return gold


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise SystemExit(f"Predictions not found: {pred_path}")

    gold_index = load_gold(args.hf_dataset, args.hf_split)

    total = 0
    exact_match = 0
    sum_recall = 0.0
    micro_correct = 0
    micro_total_gold = 0
    missing_gold = 0
    missing_pred = 0
    neo4j_times: List[float] = []
    llm_times: List[float] = []
    no_query_rows = 0

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

        # timings (support both _us and _ms fields)
        neo_us = rec.get("neo4j_time_us")
        neo_ms = rec.get("neo4j_time_ms")
        llm_us = rec.get("llm_time_us")
        llm_ms = rec.get("llm_time_ms")
        
        # Convert microseconds to milliseconds if present, else use milliseconds
        if neo_us is not None and isinstance(neo_us, (int, float)) and neo_us > 0:
            neo4j_times.append(float(neo_us) / 1000.0)
        elif isinstance(neo_ms, (int, float)) and neo_ms > 0:
            neo4j_times.append(float(neo_ms))
        else:
            # differentiate truly skipped queries (neo4j_time==0 & maybe no_evidence)
            no_query_rows += 1
            
        if llm_us is not None and isinstance(llm_us, (int, float)) and llm_us > 0:
            llm_times.append(float(llm_us) / 1000.0)
        elif isinstance(llm_ms, (int, float)) and llm_ms > 0:
            llm_times.append(float(llm_ms))

        gold_list = gold_index.get(rid)
        if gold_list is None:
            missing_gold += 1
            continue
        pred_list = to_list(rec.get("answer"))

        gold_norm: Set[str] = set(
            normalize_text(x, args.normalize) for x in gold_list if str(x).strip() != ""
        )
        pred_norm: Set[str] = set(
            normalize_text(x, args.normalize) for x in pred_list if str(x).strip() != ""
        )

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

    neo4j_times.sort()
    llm_times.sort()
    timings = {
        "neo4j_avg_ms": (sum(neo4j_times) / len(neo4j_times)) if neo4j_times else 0.0,
        "neo4j_median_ms": statistics.median(neo4j_times) if neo4j_times else 0.0,
        "neo4j_p90_ms": percentile(neo4j_times, 90) if neo4j_times else 0.0,
        "neo4j_samples": len(neo4j_times),
        "neo4j_skipped_or_zero": no_query_rows,
        "llm_avg_ms": (sum(llm_times) / len(llm_times)) if llm_times else 0.0,
        "llm_median_ms": statistics.median(llm_times) if llm_times else 0.0,
        "llm_p90_ms": percentile(llm_times, 90) if llm_times else 0.0,
        "llm_samples": len(llm_times),
    }

    metrics = {
        "samples": total,
        "exact_match_accuracy": accuracy,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "missing_gold_ids": missing_gold,
        "missing_pred_answers": missing_pred,
        **timings,
    }

    # Always print JSON to stdout for convenience
    print(json.dumps(metrics, indent=2))

    if args.output_json:
        outj = Path(args.output_json)
        outj.parent.mkdir(parents=True, exist_ok=True)
        outj.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[ok] Wrote JSON metrics to {outj}")

    if args.output_csv:
        import csv

        outc = Path(args.output_csv)
        outc.parent.mkdir(parents=True, exist_ok=True)
        # One-row CSV; create header from keys
        with outc.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"[ok] Wrote CSV metrics to {outc}")


if __name__ == "__main__":
    main()
