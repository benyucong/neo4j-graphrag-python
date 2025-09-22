from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import re
import ast
from time import perf_counter
from typing import Iterable, List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM, VLLMLLM


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch: read dataset JSONL, run 2-hop queries per path sequentially, and generate LLM answers."
    )
    p.add_argument("--input", required=True, help="JSONL dataset path")
    p.add_argument("--output", default="outputs/answers_merged.jsonl")
    p.add_argument("--max-rows", type=int, default=0, help="Process at most N rows (0 = all)")
    p.add_argument("--aggregate", choices=["min", "sum", "avg"], default="min")
    p.add_argument("--d1", type=int, default=3)
    p.add_argument("--d2", type=int, default=3)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--normalize", choices=["none", "simple", "heavy"], default="simple")
    p.add_argument("--topk", type=int, default=10, help="Max evidence rows to include in prompt")
    p.add_argument("--prompt-style", choices=["llama-inst", "chat-json"], default="llama-inst")
    p.add_argument("--uri", default="neo4j://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--provider", choices=["openai", "vllm"], default="vllm")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="sk-noop")
    p.add_argument("--metrics-output", default="", help="Optional metrics JSON path. Defaults to <output>.metrics.json")
    p.add_argument("--llm-concurrency", type=int, default=32, help="Number of concurrent LLM calls")
    return p.parse_args()


def split_path(path: str) -> Optional[Tuple[str, str]]:
    if not path:
        return None
    parts = [x.strip() for x in path.split(",")]
    parts = [x for x in parts if x]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def run_query(
    driver,
    db: str,
    start_id: str,
    wanted1: str,
    wanted2: str,
    d1_max: int,
    d2_max: int,
    limit: int,
    normalize: str,
) -> Iterable[dict]:
    if normalize == "none":
        w1_expr = "toLower($wanted1)"
        w2_expr = "toLower($wanted2)"
        p1_expr = "toLower(r1.pred)"
        p2_expr = "toLower(r2.pred)"
    elif normalize == "simple":
        w1_expr = "apoc.text.replace(toLower($wanted1),'_','')"
        w2_expr = "apoc.text.replace(toLower($wanted2),'_','')"
        p1_expr = "apoc.text.replace(toLower(r1.pred),'_','')"
        p2_expr = "apoc.text.replace(toLower(r2.pred),'_','')"
    else:  # heavy
        w1_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower($wanted1),'freebase.',''),'_',''),'.',''),'/','')"
        w2_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower($wanted2),'freebase.',''),'_',''),'.',''),'/','')"
        p1_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower(r1.pred),'_',''),'.',''),'/','')"
        p2_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower(r2.pred),'_',''),'.',''),'/','')"

    qry = (
        f"WITH {w1_expr} AS nw1, {w2_expr} AS nw2\n"
        "MATCH (s:Resource {id: $startId})-[r1:REL]-(m:Resource)-[r2:REL]-(o:Resource)\n"
        f"WITH s, r1, m, r2, o, {p1_expr} AS np1, {p2_expr} AS np2, nw1, nw2\n"
        "WITH s, r1, m, r2, o, apoc.text.distance(np1, nw1) AS d1, apoc.text.distance(np2, nw2) AS d2\n"
        "WHERE d1 <= $d1 AND d2 <= $d2\n"
        "RETURN s.id AS s, r1.pred AS p1, m.id AS m, r2.pred AS p2, o.id AS o, d1 AS d1, d2 AS d2, (d1 + d2) AS score\n"
        "ORDER BY score ASC\n"
    )
    if limit and limit > 0:
        qry += "LIMIT $limit"

    recs, _, _ = driver.execute_query(
        qry,
        {
            "startId": start_id,
            "wanted1": wanted1,
            "wanted2": wanted2,
            "d1": d1_max,
            "d2": d2_max,
            "limit": limit,
        },
        database_=db,
    )
    return recs


def get_llm(args):
    if args.provider == "openai":
        return OpenAILLM(model_name=args.model)
    return VLLMLLM(model_name=args.model, model_params=None, base_url=args.base_url, api_key=args.api_key)


def build_llama_inst_prompt(question: str, evidence: List[dict]) -> str:
    sys_txt = (
        "Based on the reasoning paths (the lower the score, the more relevant the path), answer the question. "
        "Think step by step. Return all possible answers as a Python list."
    )
    lines = []
    for r in evidence:
        s = r.get("s"); p1 = r.get("p1"); m = r.get("m"); p2 = r.get("p2"); o = r.get("o"); sc = r.get("score")
        lines.append(f"- {s} -[{p1}]-> {m} -[{p2}]-> {o} (score={sc})")
    ev_block = "\n".join(lines) if lines else "(no evidence)"
    return f"{sys_txt}\n\nreasoning paths:\n{ev_block}\n\nquestion: {question}"


def extract_answer_list(text: str) -> List[str]:
    matches = list(re.finditer(r"\[.*?\]", text, flags=re.DOTALL))
    if not matches:
        return []
    candidate = matches[-1].group(0)
    # Try JSON first
    try:
        arr = json.loads(candidate)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    # Fallback to Python list
    try:
        arr = ast.literal_eval(candidate)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        pass
    return []


def aggregate_rows(rows_by_path: Dict[str, List[dict]], aggregate: str) -> List[dict]:
    # Merge by (s, p1, m, p2, o); keep per-path scores, then aggregate to final score.
    merged: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for label, rows in rows_by_path.items():
        for r in rows:
            key = (r.get("s"), r.get("p1"), r.get("m"), r.get("p2"), r.get("o"))
            entry = merged.setdefault(
                key,
                {"s": key[0], "p1": key[1], "m": key[2], "p2": key[3], "o": key[4], "scoreA": None, "scoreB": None, "scoreC": None},
            )
            if label == "A":
                entry["scoreA"] = r.get("score")
            elif label == "B":
                entry["scoreB"] = r.get("score")
            elif label == "C":
                entry["scoreC"] = r.get("score")

    out: List[dict] = []
    for entry in merged.values():
        scores = [x for x in (entry.get("scoreA"), entry.get("scoreB"), entry.get("scoreC")) if x is not None]
        if not scores:
            continue
        if aggregate == "sum":
            score = sum(scores)
        elif aggregate == "avg":
            score = sum(scores) / float(len(scores))
        else:  # min
            score = min(scores)
        entry["score"] = score
        out.append(entry)

    out.sort(key=lambda r: r.get("score", float("inf")))
    return out


def main():
    args = parse_args()
    t_all0 = perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_output) if args.metrics_output else Path(str(outp) + ".metrics.json")

    llm = get_llm(args)
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    done = 0
    items: List[Dict[str, Any]] = []

    try:
        with inp.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if args.max_rows and done >= args.max_rows:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rid = row.get("id")
                question = row.get("question")
                q_entity = row.get("q_entity") or row.get("q_entity_id") or row.get("startId")
                if not question or not q_entity:
                    continue

                pairs: List[Tuple[str, str]] = []
                for idx in (1, 2, 3):
                    sp = split_path(row.get(f"path{idx}") or "")
                    if sp:
                        pairs.append(sp)

                if not pairs:
                    items.append({
                        "id": rid, "question": question, "q_entity": q_entity,
                        "pairs": pairs, "neo4j_time_ms": 0, "skip_llm": True
                    })
                    done += 1
                    continue

                t_q0 = perf_counter()
                rows_by_path: Dict[str, List[dict]] = {"A": [], "B": [], "C": []}
                labels = ["A", "B", "C"]
                for i, (p1, p2) in enumerate(pairs[:3]):
                    try:
                        recs = list(
                            run_query(
                                driver,
                                args.database,
                                str(q_entity),
                                p1,
                                p2,
                                args.d1,
                                args.d2,
                                args.limit,
                                args.normalize,
                            )
                        )
                    except Exception:
                        recs = []
                    rows_by_path[labels[i]] = recs

                neo4j_time_ms = int((perf_counter() - t_q0) * 1000)

                evidence_all = aggregate_rows(rows_by_path, args.aggregate)
                if not evidence_all:
                    items.append({
                        "id": rid, "question": question, "q_entity": q_entity,
                        "pairs": pairs, "neo4j_time_ms": neo4j_time_ms, "skip_llm": True
                    })
                    done += 1
                    continue

                top = evidence_all[: args.topk]
                if args.prompt_style == "llama-inst":
                    prompt = build_llama_inst_prompt(question, top)
                else:
                    prompt = json.dumps({"question": question, "evidence": top}, ensure_ascii=False)

                items.append({
                    "id": rid, "question": question, "q_entity": q_entity,
                    "pairs": pairs, "neo4j_time_ms": neo4j_time_ms,
                    "prompt": prompt, "skip_llm": False
                })
                done += 1
    finally:
        driver.close()

    # LLM fan-out (concurrent)
    def llm_task(index: int, item: dict):
        if item.get("skip_llm"):
            return index, {"answer": [], "llm_time_ms": 0}
        prompt = item["prompt"]
        rid = item.get("id")
        print(f"\n[Prompt to LLM][id={rid}]:\n{prompt}")
        t0 = perf_counter()
        try:
            resp = llm.invoke(prompt)
            raw = resp.content.strip()
            ans = extract_answer_list(raw)
            return index, {"answer": ans, "llm_time_ms": int((perf_counter() - t0) * 1000)}
        except Exception as e:
            return index, {"answer": [f"llm_error: {e}"], "llm_time_ms": int((perf_counter() - t0) * 1000)}

    llm_results: List[Optional[dict]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max(1, int(args.llm_concurrency))) as ex:
        futures = {ex.submit(llm_task, i, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            idx, res = fut.result()
            llm_results[idx] = res

    for i in range(len(llm_results)):
        if llm_results[i] is None:
            llm_results[i] = {"answer": [], "llm_time_ms": 0}

    # Write outputs in order
    with outp.open("w", encoding="utf-8") as f_out:
        for i, item in enumerate(items):
            if item.get("skip_llm"):
                out = {
                    "id": item["id"], "question": item["question"], "q_entity": item["q_entity"],
                    "paths": item["pairs"], "answer": [], "no_evidence": True,
                    "neo4j_time_ms": item["neo4j_time_ms"], "llm_time_ms": 0
                }
            else:
                out = {
                    "id": item["id"], "question": item["question"], "q_entity": item["q_entity"],
                    "paths": item["pairs"],
                    "answer": llm_results[i]["answer"],
                    "neo4j_time_ms": item["neo4j_time_ms"],
                    "llm_time_ms": llm_results[i]["llm_time_ms"]
                }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Metrics
    elapsed_s = max(1e-9, perf_counter() - t_all0)
    end_iso = datetime.now(timezone.utc).isoformat()
    throughput = done / elapsed_s
    print(f"[ok] Wrote {done} rows to {outp}")

    metrics = {
        "dataset_path": str(inp),
        "output_path": str(outp),
        "rows_processed": done,
        "start_time": start_iso,
        "end_time": end_iso,
        "elapsed_seconds": elapsed_s,
        "throughput_rows_per_sec": throughput,
        "aggregate": args.aggregate,
        "d1": args.d1,
        "d2": args.d2,
        "limit": args.limit,
        "normalize": args.normalize,
        "topk": args.topk,
        "provider": args.provider,
        "model": args.model,
        "uri": args.uri,
        "database": args.database,
    }
    metrics_path = Path(args.metrics_output) if args.metrics_output else Path(str(outp) + ".metrics.json")
    try:
        with metrics_path.open("w", encoding="utf-8") as f_m:
            f_m.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"[ok] Metrics written to {metrics_path}")
    except Exception as e:
        print(f"[warn] Failed to write metrics to {metrics_path}: {e}")


if __name__ == "__main__":
    main()
