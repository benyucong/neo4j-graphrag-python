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
        description="Batch: read dataset JSONL, run merged 3-hop queries, and generate LLM answers."
    )
    p.add_argument("--input", required=True, help="JSONL dataset path")
    p.add_argument("--output", default="outputs/answers_merged_3hop.jsonl")
    p.add_argument("--max-rows", type=int, default=0, help="Process at most N rows (0 = all)")
    p.add_argument("--aggregate", choices=["min", "sum", "avg"], default="min")
    p.add_argument("--d1", type=int, default=3, help="Max distance for 1st hop")
    p.add_argument("--d2", type=int, default=3, help="Max distance for 2nd hop")
    p.add_argument("--d3", type=int, default=3, help="Max distance for 3rd hop")
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


def split_path_3hop(path: str) -> Optional[Tuple[str, str, str]]:
    """Parse a 3-hop path as 'pred1, pred2, pred3'."""
    if not path:
        return None
    parts = [x.strip() for x in path.split(",")]
    parts = [x for x in parts if x]
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def run_query_merged_3hop(
    driver,
    db: str,
    start_id: str,
    triples: List[Tuple[str, str, str]],
    d1_max: int,
    d2_max: int,
    d3_max: int,
    limit: int,
    normalize: str,
    aggregate: str,
) -> Iterable[dict]:
    """
    Run a merged 3-hop query across up to 3 paths.
    Each path is (pred1, pred2, pred3).
    Returns rows: s -[p1]-> m1 -[p2]-> m2 -[p3]-> o
    with scoreA, scoreB, scoreC, combined score.
    """
    # Pad to 3 paths
    t = triples[:3] + [("", "", "")] * (3 - len(triples))
    (w1a, w2a, w3a), (w1b, w2b, w3b), (w1c, w2c, w3c) = t
    hasA = bool(w1a and w2a and w3a)
    hasB = bool(w1b and w2b and w3b)
    hasC = bool(w1c and w2c and w3c)

    # Build normalization expressions
    if normalize == "none":
        w1a_expr = "toLower($w1a)"; w2a_expr = "toLower($w2a)"; w3a_expr = "toLower($w3a)"
        w1b_expr = "toLower($w1b)"; w2b_expr = "toLower($w2b)"; w3b_expr = "toLower($w3b)"
        w1c_expr = "toLower($w1c)"; w2c_expr = "toLower($w2c)"; w3c_expr = "toLower($w3c)"
        p1_expr = "toLower(r1.pred)"; p2_expr = "toLower(r2.pred)"; p3_expr = "toLower(r3.pred)"
    elif normalize == "simple":
        w1a_expr = "apoc.text.replace(toLower($w1a),'_','')"; w2a_expr = "apoc.text.replace(toLower($w2a),'_','')"; w3a_expr = "apoc.text.replace(toLower($w3a),'_','')"
        w1b_expr = "apoc.text.replace(toLower($w1b),'_','')"; w2b_expr = "apoc.text.replace(toLower($w2b),'_','')"; w3b_expr = "apoc.text.replace(toLower($w3b),'_','')"
        w1c_expr = "apoc.text.replace(toLower($w1c),'_','')"; w2c_expr = "apoc.text.replace(toLower($w2c),'_','')"; w3c_expr = "apoc.text.replace(toLower($w3c),'_','')"
        p1_expr = "apoc.text.replace(toLower(r1.pred),'_','')"; p2_expr = "apoc.text.replace(toLower(r2.pred),'_','')"; p3_expr = "apoc.text.replace(toLower(r3.pred),'_','')"
    else:  # heavy
        def heavy(arg: str) -> str:
            return (
                f"apoc.text.replace(apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower({arg}),'freebase.',''),'_',''),'.',''),'/','')"
            )
        w1a_expr = heavy("$w1a"); w2a_expr = heavy("$w2a"); w3a_expr = heavy("$w3a")
        w1b_expr = heavy("$w1b"); w2b_expr = heavy("$w2b"); w3b_expr = heavy("$w3b")
        w1c_expr = heavy("$w1c"); w2c_expr = heavy("$w2c"); w3c_expr = heavy("$w3c")
        p1_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower(r1.pred),'_',''),'.',''),'/','')"
        p2_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower(r2.pred),'_',''),'.',''),'/','')"
        p3_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower(r3.pred),'_',''),'.',''),'/','')"

    qry = f"""
    // Precompute normalized keywords once at top level
    WITH
      {w1a_expr} AS w1a, {w2a_expr} AS w2a, {w3a_expr} AS w3a,
      {w1b_expr} AS w1b, {w2b_expr} AS w2b, {w3b_expr} AS w3b,
      {w1c_expr} AS w1c, {w2c_expr} AS w2c, {w3c_expr} AS w3c
    MATCH (s:Resource {{id: $startId}})

    // ---------- First hop: s -[r1]- m1 (global prune to top $d1) ----------
    CALL {{
      WITH s, w1a, w1b, w1c
      MATCH (s)-[r1:REL]-(m1:Resource)
      WITH r1, m1, {p1_expr} AS np1, w1a, w1b, w1c
      WITH r1, m1,
           CASE WHEN $hasA THEN apoc.text.distance(np1, w1a) ELSE NULL END AS a1,
           CASE WHEN $hasB THEN apoc.text.distance(np1, w1b) ELSE NULL END AS b1,
           CASE WHEN $hasC THEN apoc.text.distance(np1, w1c) ELSE NULL END AS c1
      WHERE ( ($hasA AND a1 IS NOT NULL AND a1 <= $d1)
           OR ($hasB AND b1 IS NOT NULL AND b1 <= $d1)
           OR ($hasC AND c1 IS NOT NULL AND c1 <= $d1) )
      WITH r1, m1, a1, b1, c1,
           reduce(minv=1e9, x IN [a1,b1,c1] | CASE WHEN x IS NULL OR x>minv THEN minv ELSE x END) AS rank1
      ORDER BY rank1 ASC
      LIMIT $d1
      RETURN r1, m1, a1, b1, c1
    }}

    // ---------- Second hop: m1 -[r2]- m2 (per-m1 prune to top $d2) ----------
    CALL {{
      WITH m1, w2a, w2b, w2c
      MATCH (m1)-[r2:REL]-(m2:Resource)
      WITH r2, m2, {p2_expr} AS np2, w2a, w2b, w2c
      WITH r2, m2,
           CASE WHEN $hasA THEN apoc.text.distance(np2, w2a) ELSE NULL END AS a2,
           CASE WHEN $hasB THEN apoc.text.distance(np2, w2b) ELSE NULL END AS b2,
           CASE WHEN $hasC THEN apoc.text.distance(np2, w2c) ELSE NULL END AS c2
      WHERE ( ($hasA AND a2 IS NOT NULL AND a2 <= $d2)
           OR ($hasB AND b2 IS NOT NULL AND b2 <= $d2)
           OR ($hasC AND c2 IS NOT NULL AND c2 <= $d2) )
      WITH r2, m2, a2, b2, c2,
           reduce(minv=1e9, x IN [a2,b2,c2] | CASE WHEN x IS NULL OR x>minv THEN minv ELSE x END) AS rank2
      ORDER BY rank2 ASC
      LIMIT $d2
      RETURN r2, m2, a2, b2, c2
    }}

    // ---------- Third hop: m2 -[r3]- o (per-m2 prune to top $d3) ----------
    CALL {{
      WITH m2, w3a, w3b, w3c
      MATCH (m2)-[r3:REL]-(o:Resource)
      WITH r3, o, {p3_expr} AS np3, w3a, w3b, w3c
      WITH r3, o,
           CASE WHEN $hasA THEN apoc.text.distance(np3, w3a) ELSE NULL END AS a3,
           CASE WHEN $hasB THEN apoc.text.distance(np3, w3b) ELSE NULL END AS b3,
           CASE WHEN $hasC THEN apoc.text.distance(np3, w3c) ELSE NULL END AS c3
      WHERE ( ($hasA AND a3 IS NOT NULL AND a3 <= $d3)
           OR ($hasB AND b3 IS NOT NULL AND b3 <= $d3)
           OR ($hasC AND c3 IS NOT NULL AND c3 <= $d3) )
      WITH r3, o, a3, b3, c3,
           reduce(minv=1e9, x IN [a3,b3,c3] | CASE WHEN x IS NULL OR x>minv THEN minv ELSE x END) AS rank3
      ORDER BY rank3 ASC
      LIMIT $d3
      RETURN r3, o, a3, b3, c3
    }}

    // ---------- Combine hops and score ----------
    WITH s, r1, m1, r2, m2, r3, o,
         a1, b1, c1, a2, b2, c2, a3, b3, c3,
         CASE WHEN $hasA AND a1 IS NOT NULL AND a2 IS NOT NULL AND a3 IS NOT NULL THEN a1 + a2 + a3 ELSE NULL END AS scoreA,
         CASE WHEN $hasB AND b1 IS NOT NULL AND b2 IS NOT NULL AND b3 IS NOT NULL THEN b1 + b2 + b3 ELSE NULL END AS scoreB,
         CASE WHEN $hasC AND c1 IS NOT NULL AND c2 IS NOT NULL AND c3 IS NOT NULL THEN c1 + c2 + c3 ELSE NULL END AS scoreC
    WITH s, r1, m1, r2, m2, r3, o, scoreA, scoreB, scoreC,
         [x IN [scoreA, scoreB, scoreC] WHERE x IS NOT NULL] AS scores

    WITH s, r1, m1, r2, m2, r3, o, scoreA, scoreB, scoreC, scores,
         CASE $aggregate
           WHEN 'sum' THEN reduce(acc=0.0, x IN scores | acc + x)
           WHEN 'avg' THEN (reduce(acc=0.0, x IN scores | acc + x) / toFloat(size(scores)))
           ELSE reduce(minVal=1e9, x IN scores | CASE WHEN x < minVal THEN x ELSE minVal END)
         END AS score

    RETURN s.id AS s, r1.pred AS p1, m1.id AS m1, r2.pred AS p2, m2.id AS m2, r3.pred AS p3, o.id AS o,
           scoreA, scoreB, scoreC, score
    ORDER BY score ASC
    {"LIMIT $limit" if limit and limit > 0 else ""}
    """

    recs, _, _ = driver.execute_query(
        qry,
        {
            "startId": start_id,
            "w1a": w1a or "", "w2a": w2a or "", "w3a": w3a or "",
            "w1b": w1b or "", "w2b": w2b or "", "w3b": w3b or "",
            "w1c": w1c or "", "w2c": w2c or "", "w3c": w3c or "",
            "hasA": hasA, "hasB": hasB, "hasC": hasC,
            "d1": int(d1_max), "d2": int(d2_max), "d3": int(d3_max),
            "limit": int(limit or 0),
            "aggregate": aggregate,
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
        "Based on the reasoning paths (the lower score, the more relevant the path), please answer the given question. "
        "Please think it step by step. "
        "Please keep the answer as simple as possible and return all the possible answers as a python list "
    )
    lines = []
    for r in evidence:
        s = r.get("s"); p1 = r.get("p1"); m1 = r.get("m1"); p2 = r.get("p2"); m2 = r.get("m2"); p3 = r.get("p3"); o = r.get("o"); sc = r.get("score")
        lines.append(f"- {s} -[{p1}]-> {m1} -[{p2}]-> {m2} -[{p3}]-> {o} (score={sc})")
    ev_block = "\n".join(lines) if lines else "(no evidence)"
    return (
        f"{sys_txt}\n\nreasoning paths:\n{ev_block}\n\nquestion: {question}"
    )


def extract_answer_list(text: str) -> List[str]:
    matches = list(re.finditer(r"\[.*?\]", text, flags=re.DOTALL))
    if not matches:
        return []
    candidate = matches[-1].group(0)
    # Strict JSON first
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

    # Init LLM and Neo4j driver once
    llm = get_llm(args)
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))

    done = 0
    items: List[Dict[str, Any]] = []

    try:
        with inp.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                if args.max_rows and args.max_rows > 0 and done >= args.max_rows:
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

                triples: List[Tuple[str, str, str]] = []
                for idx in (1, 2, 3):
                    path = row.get(f"path{idx}") or ""
                    sp = split_path_3hop(path)
                    if sp:
                        triples.append(sp)

                # No usable paths → record a no-evidence item
                if not triples:
                    items.append({
                        "id": rid, "question": question, "q_entity": q_entity,
                        "triples": triples, "skip_llm": True, "neo4j_time_us": 0
                    })
                    done += 1
                    continue

                t_q0 = perf_counter()
                try:
                    evidence = list(
                        run_query_merged_3hop(
                            driver, args.database, str(q_entity), triples,
                            args.d1, args.d2, args.d3, args.limit, args.normalize, args.aggregate
                        )
                    )
                    neo4j_time_us = int((perf_counter() - t_q0) * 1_000_000)
                except Exception as e:
                    neo4j_time_us = int((perf_counter() - t_q0) * 1_000_000)
                    items.append({
                        "id": rid, "question": question, "q_entity": q_entity,
                        "triples": triples, "error": str(e), "neo4j_time_us": neo4j_time_us
                    })
                    done += 1
                    continue

                if not evidence:
                    items.append({
                        "id": rid, "question": question, "q_entity": q_entity,
                        "triples": triples, "skip_llm": True, "neo4j_time_us": neo4j_time_us
                    })
                    done += 1
                    continue

                top = evidence[: args.topk]
                if args.prompt_style == "llama-inst":
                    prompt = build_llama_inst_prompt(question, top)
                else:
                    # simple JSON chat style as fallback (single string input)
                    prompt = json.dumps({"question": question, "evidence": top}, ensure_ascii=False)

                items.append({
                    "id": rid, "question": question, "q_entity": q_entity,
                    "triples": triples, "prompt": prompt, "neo4j_time_us": neo4j_time_us
                })
                done += 1
    finally:
        driver.close()

    # Stage 2: run LLM concurrently where needed
    def llm_task(index: int, item: dict):
        if item.get("skip_llm") or "prompt" not in item:
            return index, {"answer": [], "llm_time_us": 0}
        prompt = item["prompt"]
        rid = item.get("id")
        print(f"\n[Prompt to LLM][id={rid}]:\n{prompt}")
        t0 = perf_counter()
        try:
            resp = llm.invoke(prompt)
            raw = resp.content.strip()
            print("[LLM raw]:", raw)
            ans = extract_answer_list(raw)
            return index, {"answer": ans, "llm_time_us": int((perf_counter() - t0) * 1_000_000)}
        except Exception as e:
            return index, {"answer": [f"llm_error: {e}"], "llm_time_us": int((perf_counter() - t0) * 1_000_000)}

    llm_results: List[Optional[dict]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max(1, int(args.llm_concurrency))) as ex:
        futures = {ex.submit(llm_task, i, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            idx, res = fut.result()
            llm_results[idx] = res

    for i in range(len(llm_results)):
        if llm_results[i] is None:
            llm_results[i] = {"answer": [], "llm_time_us": 0}

    # Stage 3: write outputs in order
    with outp.open("w", encoding="utf-8") as f_out:
        for i, item in enumerate(items):
            if "error" in item:
                out = {
                    "id": item["id"], "question": item["question"], "q_entity": item["q_entity"],
                    "paths": item["triples"], "error": item["error"], "neo4j_time_us": item["neo4j_time_us"]
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            if item.get("skip_llm"):
                out = {
                    "id": item["id"], "question": item["question"], "q_entity": item["q_entity"],
                    "paths": item["triples"], "answer": [],
                    "no_evidence": True,
                    "neo4j_time_us": item["neo4j_time_us"], "llm_time_us": 0
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            llm_res = llm_results[i]
            out = {
                "id": item["id"], "question": item["question"], "q_entity": item["q_entity"],
                "paths": item["triples"], "answer": llm_res["answer"],
                "neo4j_time_us": item["neo4j_time_us"], "llm_time_us": llm_res["llm_time_us"]
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    elapsed_s = max(1e-9, perf_counter() - t_all0)
    end_iso = datetime.now(timezone.utc).isoformat()
    throughput = done / elapsed_s
    print(f"[ok] Wrote {done} rows to {outp}")

    # Write metrics JSON
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
        "d3": args.d3,
        "limit": args.limit,
        "normalize": args.normalize,
        "topk": args.topk,
        "provider": args.provider,
        "model": args.model,
        "uri": args.uri,
        "database": args.database,
    }
    try:
        with metrics_path.open("w", encoding="utf-8") as f_m:
            f_m.write(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"[ok] Metrics written to {metrics_path}")
    except Exception as e:
        print(f"[warn] Failed to write metrics to {metrics_path}: {e}")


if __name__ == "__main__":
    main()
