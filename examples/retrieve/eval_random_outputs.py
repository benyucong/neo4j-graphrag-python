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
        description="Analyze random_outputs directory: aggregate per-limit files to CSV with metrics."
    )
    p.add_argument(
        "--input-dir", default="random_outputs", help="Directory containing answers_sequential_random_limitX.jsonl files"
    )
    p.add_argument(
        "--glob-pattern", default="answers_sequential_random_limit*.jsonl", help="Glob pattern within input directory"
    )
    p.add_argument(
        "--hf-dataset", default="Benyucong/vanilla-hop2", help="HF dataset with gold a_entity field"
    )
    p.add_argument("--hf-split", default="test")
    p.add_argument(
        "--normalize", choices=["none", "simple"], default="none", help="Normalization mode for answer comparison"
    )
    p.add_argument(
        "--output-csv", default="random_outputs/random_limit_sweep_metrics.csv", help="Destination CSV file"
    )
    p.add_argument(
        "--include-timing-percentiles", action="store_true", help="Include p50/p95 timing percentiles"
    )
    return p.parse_args()


def normalize_text(s: str, mode: str) -> str:
    """Normalize text for comparison."""
    t = s.strip().strip('"').strip("'").lower()
    if mode == "simple":
        t = t.replace("_", " ")
    return " ".join(t.split())


def to_list(obj: Any) -> List[str]:
    """Convert various formats to list of strings."""
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
    """Load ground truth from HuggingFace dataset."""
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
    """Calculate percentile of a sorted list."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(values_sorted):
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


MetricResult = Dict[str, Any]


def evaluate_file(
    path: Path, gold_index: Dict[int, List[str]], normalize: str, include_percentiles: bool
) -> Tuple[MetricResult, int, float, float, Dict[str, float]]:
    """
    Return (metrics, rows, avg_neo4j_ms, avg_llm_ms, timing_percentiles).
    timing_percentiles contains p50/p95 for neo4j and llm times if requested.
    """
    total = 0
    exact_match = 0
    sum_recall = 0.0
    micro_correct = 0
    micro_total_gold = 0
    missing_gold = 0
    missing_pred = 0
    neo4j_times: List[float] = []
    llm_times: List[float] = []
    neo4j_zero_count = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Collect timing data - handle both microseconds and milliseconds for backward compatibility
        if "neo4j_time_us" in rec:
            neo4j_val_ms = (rec.get("neo4j_time_us", 0) or 0) / 1000.0  # Convert μs to ms
        elif "neo4j_time_ms" in rec:
            neo4j_val_ms = float(rec.get("neo4j_time_ms", 0) or 0)
        else:
            neo4j_val_ms = 0.0
        
        if neo4j_val_ms > 0 or "neo4j_time_us" in rec or "neo4j_time_ms" in rec:
            neo4j_times.append(neo4j_val_ms)
            if neo4j_val_ms == 0:
                neo4j_zero_count += 1
        
        # Handle both llm_time_us and llm_time_ms
        if "llm_time_us" in rec:
            llm_val_ms = (rec.get("llm_time_us", 0) or 0) / 1000.0  # Convert μs to ms
        elif "llm_time_ms" in rec:
            llm_val_ms = float(rec.get("llm_time_ms", 0) or 0)
        else:
            llm_val_ms = 0.0
        
        if llm_val_ms > 0 or "llm_time_us" in rec or "llm_time_ms" in rec:
            llm_times.append(llm_val_ms)

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

    # Calculate metrics
    accuracy = (exact_match / total) if total else 0.0
    recall_macro = (sum_recall / total) if total else 0.0
    recall_micro = (micro_correct / micro_total_gold) if micro_total_gold else 0.0
    avg_neo4j = (sum(neo4j_times) / len(neo4j_times)) if neo4j_times else 0.0
    avg_llm = (sum(llm_times) / len(llm_times)) if llm_times else 0.0
    max_neo4j = max(neo4j_times) if neo4j_times else 0.0

    timing_percentiles: Dict[str, float] = {}
    if include_percentiles:
        timing_percentiles["neo4j_p50"] = percentile(neo4j_times, 0.50) if neo4j_times else 0.0
        timing_percentiles["neo4j_p95"] = percentile(neo4j_times, 0.95) if neo4j_times else 0.0
        timing_percentiles["llm_p50"] = percentile(llm_times, 0.50) if llm_times else 0.0
        timing_percentiles["llm_p95"] = percentile(llm_times, 0.95) if llm_times else 0.0

    metrics: MetricResult = {
        "samples": total,
        "exact_match_accuracy": accuracy,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "missing_gold_ids": missing_gold,
        "missing_pred_answers": missing_pred,
        "neo4j_zero_count": neo4j_zero_count,
        "neo4j_max_time_ms": max_neo4j,
    }
    
    return metrics, total, avg_neo4j, avg_llm, timing_percentiles


def main() -> None:
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    
    # Find all matching files
    files = sorted(input_dir.glob(args.glob_pattern))
    if not files:
        raise SystemExit(f"No files matched pattern '{args.glob_pattern}' in {input_dir}")
    
    # Extract numeric limit from filename
    limit_re = re.compile(r"limit(\d+)\.jsonl$")
    selected: List[Tuple[int, Path]] = []
    for f in files:
        m = limit_re.search(f.name)
        if m:
            selected.append((int(m.group(1)), f))
    
    if not selected:
        raise SystemExit(f"No files with pattern '*limit<number>.jsonl' found in {input_dir}")
    
    selected.sort(key=lambda x: x[0])
    
    print(f"[info] Found {len(selected)} limit files in {input_dir}")
    print(f"[info] Files: {[f.name for _, f in selected]}")
    
    # Load ground truth
    print(f"[info] Loading ground truth from {args.hf_dataset} ({args.hf_split} split)...")
    gold_index = load_gold(args.hf_dataset, args.hf_split)
    print(f"[info] Loaded {len(gold_index)} ground truth samples")
    
    # Prepare output CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "limit",
        "samples",
        "avg_neo4j_time_ms",
        "neo4j_max_time_ms",
        "avg_llm_time_ms",
    ]
    
    if args.include_timing_percentiles:
        fieldnames.extend([
            "neo4j_p50_ms",
            "neo4j_p95_ms",
            "llm_p50_ms",
            "llm_p95_ms",
        ])
    
    fieldnames.extend([
        "exact_match_accuracy",
        "recall_macro",
        "recall_micro",
        "missing_pred_answers",
        "missing_gold_ids",
        "neo4j_zero_count",
    ])
    
    with out_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        
        for limit, path in selected:
            print(f"[eval] Processing limit={limit}: {path.name}")
            metrics, samples, avg_neo4j, avg_llm, timing_pcts = evaluate_file(
                path, gold_index, args.normalize, args.include_timing_percentiles
            )
            
            max_neo4j = metrics["neo4j_max_time_ms"]

            row = {
                "limit": limit,
                "samples": samples,
                "avg_neo4j_time_ms": round(avg_neo4j, 2),
                "neo4j_max_time_ms": round(max_neo4j, 2),
                "avg_llm_time_ms": round(avg_llm, 2),
                "exact_match_accuracy": round(metrics["exact_match_accuracy"], 6),
                "recall_macro": round(metrics["recall_macro"], 6),
                "recall_micro": round(metrics["recall_micro"], 6),
                "missing_pred_answers": metrics["missing_pred_answers"],
                "missing_gold_ids": metrics["missing_gold_ids"],
                "neo4j_zero_count": metrics["neo4j_zero_count"],
            }
            
            if args.include_timing_percentiles:
                row.update({
                    "neo4j_p50_ms": round(timing_pcts.get("neo4j_p50", 0.0), 2),
                    "neo4j_p95_ms": round(timing_pcts.get("neo4j_p95", 0.0), 2),
                    "llm_p50_ms": round(timing_pcts.get("llm_p50", 0.0), 2),
                    "llm_p95_ms": round(timing_pcts.get("llm_p95", 0.0), 2),
                })
            
            writer.writerow(row)
            
            # Print summary
            print(
                f"  samples={samples}, accuracy={metrics['exact_match_accuracy']:.4f}, "
                f"recall_macro={metrics['recall_macro']:.4f}, "
                f"avg_neo4j={avg_neo4j:.2f}ms, max_neo4j={max_neo4j:.2f}ms, "
                f"avg_llm={avg_llm:.2f}ms"
            )
    
    print(f"\n[ok] Wrote sweep metrics to {out_path}")
    print(f"[info] Total limits analyzed: {len(selected)}")


if __name__ == "__main__":
    main()
