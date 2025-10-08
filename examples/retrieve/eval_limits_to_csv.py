from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-limit prediction files (answers_merged_sharp_limitX.jsonl) to CSV with avg Neo4j time and accuracy metrics."
    )
    p.add_argument(
        "--glob", default="outputs/answers_merged_sharp_limit*.jsonl", help="Glob pattern for prediction JSONL files"
    )
    p.add_argument(
        "--hf-dataset", default="Benyucong/vanilla-hop2", help="HF dataset with gold a_entity field"
    )
    p.add_argument("--hf-split", default="test")
    p.add_argument(
        "--normalize", choices=["none", "simple"], default="none", help="Normalization mode like other evaluator"
    )
    p.add_argument(
        "--output-csv", default="outputs/limit_sweep_metrics.csv", help="Destination CSV file"
    )
    p.add_argument(
        "--include-llm", action="store_true", help="Also compute average llm_time_ms (default off)"
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
        try:  # JSON
            val = json.loads(txt)
            if isinstance(val, list):
                return [str(x) for x in val]
        except Exception:
            pass
        try:  # Python literal list
            val = ast.literal_eval(txt)
            if isinstance(val, list):
                return [str(x) for x in val]
        except Exception:
            pass
        return [txt]
    return [str(obj)]


def load_gold(hf_dataset: str, hf_split: str) -> Dict[int, List[str]]:
    ds = load_dataset(hf_dataset, split=hf_split)
    out: Dict[int, List[str]] = {}
    for row in ds:
        try:
            rid = int(row.get("id"))
        except Exception:
            continue
        out[rid] = to_list(row.get("a_entity"))
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(values_sorted):
        return float(values_sorted[f])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


MetricResult = Dict[str, Any]


def evaluate_file(
    path: Path, gold_index: Dict[int, List[str]], normalize: str
) -> Tuple[MetricResult, int, float, float, float, float]:
    """Return (metrics, rows, avg_neo4j_ms, avg_llm_ms, max_neo4j_ms, p95_neo4j_ms)."""
    total = 0
    exact_match = 0
    sum_recall = 0.0
    micro_correct = 0
    micro_total_gold = 0
    missing_gold = 0
    missing_pred = 0
    neo4j_sum = 0.0
    neo4j_count = 0
    neo4j_max = 0.0
    neo4j_times: List[float] = []
    llm_sum = 0.0
    llm_count = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Handle both neo4j_time_us (microseconds) and neo4j_time_ms (milliseconds) for backward compatibility
        if "neo4j_time_us" in rec:
            neo4j_val_ms = (rec.get("neo4j_time_us", 0) or 0) / 1000.0  # Convert μs to ms
        elif "neo4j_time_ms" in rec:
            neo4j_val_ms = float(rec.get("neo4j_time_ms", 0) or 0)
        else:
            neo4j_val_ms = 0.0
        
        if neo4j_val_ms > 0 or "neo4j_time_us" in rec or "neo4j_time_ms" in rec:
            neo4j_sum += neo4j_val_ms
            neo4j_count += 1
            if neo4j_val_ms > neo4j_max:
                neo4j_max = neo4j_val_ms
            neo4j_times.append(neo4j_val_ms)
        
        # Handle both llm_time_us and llm_time_ms for backward compatibility
        if "llm_time_us" in rec:
            llm_val_ms = (rec.get("llm_time_us", 0) or 0) / 1000.0  # Convert μs to ms
        elif "llm_time_ms" in rec:
            llm_val_ms = float(rec.get("llm_time_ms", 0) or 0)
        else:
            llm_val_ms = 0.0
        
        if llm_val_ms > 0 or "llm_time_us" in rec or "llm_time_ms" in rec:
            llm_sum += llm_val_ms
            llm_count += 1

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
        gold_norm: Set[str] = set(
            normalize_text(x, normalize) for x in gold_list if str(x).strip() != ""
        )
        pred_norm: Set[str] = set(
            normalize_text(x, normalize) for x in pred_list if str(x).strip() != ""
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
    avg_neo4j = (neo4j_sum / neo4j_count) if neo4j_count else 0.0
    avg_llm = (llm_sum / llm_count) if llm_count else 0.0
    max_neo4j = neo4j_max if neo4j_count else 0.0
    p95_neo4j = percentile(neo4j_times, 0.90) if neo4j_times else 0.0
    metrics: MetricResult = {
        "samples": total,
        "exact_match_accuracy": accuracy,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "missing_gold_ids": missing_gold,
        "missing_pred_answers": missing_pred,
        "neo4j_max_time_ms": max_neo4j,
        "neo4j_p95_time_ms": p95_neo4j,
    }
    return metrics, total, avg_neo4j, avg_llm, max_neo4j, p95_neo4j


def main() -> None:
    args = parse_args()
    files = sorted(Path('.').glob(args.glob.replace('outputs/', 'outputs/')))  # simple glob
    # Fallback using Path.glob directly (above may be redundant but safe)
    if not files:
        files = sorted(Path('.').glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched glob: {args.glob}")

    # Extract numeric limit; keep only files with pattern limit<number>.jsonl
    limit_re = re.compile(r"limit(\d+)\.jsonl$")
    selected: List[Tuple[int, Path]] = []
    for f in files:
        m = limit_re.search(f.name)
        if m:
            selected.append((int(m.group(1)), f))
    if not selected:
        raise SystemExit("No prediction files with pattern '*limit<number>.jsonl' found")
    selected.sort(key=lambda x: x[0])

    print(f"[info] Evaluating {len(selected)} files: {[f.name for _, f in selected]}")
    gold_index = load_gold(args.hf_dataset, args.hf_split)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "limit",
        "samples",
        "avg_neo4j_time_ms",
        "neo4j_max_time_ms",
        "neo4j_p90_time_ms",
        "avg_llm_time_ms",
        "exact_match_accuracy",
        "recall_macro",
        "recall_micro",
        "missing_pred_answers",
        "missing_gold_ids",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for limit, path in selected:
            metrics, samples, avg_neo4j, avg_llm, max_neo4j, p95_neo4j = evaluate_file(
                path, gold_index, args.normalize
            )
            row = {
                "limit": limit,
                "samples": samples,
                "avg_neo4j_time_ms": round(avg_neo4j, 2),
                "neo4j_max_time_ms": round(max_neo4j, 2),
                "neo4j_p95_time_ms": round(p95_neo4j, 2),
                "avg_llm_time_ms": round(avg_llm, 2),
                "exact_match_accuracy": round(metrics["exact_match_accuracy"], 6),
                "recall_macro": round(metrics["recall_macro"], 6),
                "recall_micro": round(metrics["recall_micro"], 6),
                "missing_pred_answers": metrics["missing_pred_answers"],
                "missing_gold_ids": metrics["missing_gold_ids"],
            }
            if not args.include_llm:
                # Optionally blank out avg_llm_time if user doesn't care
                pass
            writer.writerow(row)
    print(f"[ok] Wrote sweep metrics to {out_path}")


if __name__ == "__main__":
    main()
