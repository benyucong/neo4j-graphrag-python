from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM, VLLMLLM


def parse_args():
    p = argparse.ArgumentParser(description="Batch: read dataset JSONL, run merged 2-hop queries, and generate LLM answers.")
    p.add_argument("--input", required=True, help="JSONL dataset path")
    p.add_argument("--output", default="outputs/answers_merged.jsonl")
    p.add_argument("--max-rows", type=int, default=50, help="Process at most N rows (0 = all)")
    p.add_argument("--aggregate", choices=["min","sum","avg"], default="min")
    p.add_argument("--d1", type=int, default=3)
    p.add_argument("--d2", type=int, default=3)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--normalize", choices=["none","simple","heavy"], default="simple")
    p.add_argument("--topk", type=int, default=10, help="Max evidence rows to include in prompt")
    p.add_argument("--prompt-style", choices=["llama-inst","chat-json"], default="llama-inst")
    p.add_argument("--uri", default="neo4j://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--provider", choices=["openai","vllm"], default="vllm")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="sk-noop")
    return p.parse_args()


def split_path(path: str) -> Optional[Tuple[str, str]]:
    if not path:
        return None
    parts = [x.strip() for x in path.split(",")]
    parts = [x for x in parts if x]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def run_query_merged(
    uri: str,
    user: str,
    password: str,
    db: str,
    start_id: str,
    pairs: List[Tuple[str, str]],
    d1_max: int,
    d2_max: int,
    limit: int,
    normalize: str,
    aggregate: str,
) -> Iterable[dict]:
    p = pairs[:3] + [("", "")] * (3 - len(pairs))
    (w1a, w2a), (w1b, w2b), (w1c, w2c) = p
    hasA = bool(w1a and w2a)
    hasB = bool(w1b and w2b)
    hasC = bool(w1c and w2c)

    if normalize == "none":
        w1a_expr = "toLower($w1a)"; w2a_expr = "toLower($w2a)"
        w1b_expr = "toLower($w1b)"; w2b_expr = "toLower($w2b)"
        w1c_expr = "toLower($w1c)"; w2c_expr = "toLower($w2c)"
        p1_expr = "toLower(r1.pred)"; p2_expr = "toLower(r2.pred)"
    elif normalize == "simple":
        w1a_expr = "apoc.text.replace(toLower($w1a),'_','')"; w2a_expr = "apoc.text.replace(toLower($w2a),'_','')"
        w1b_expr = "apoc.text.replace(toLower($w1b),'_','')"; w2b_expr = "apoc.text.replace(toLower($w2b),'_','')"
        w1c_expr = "apoc.text.replace(toLower($w1c),'_','')"; w2c_expr = "apoc.text.replace(toLower($w2c),'_','')"
        p1_expr = "apoc.text.replace(toLower(r1.pred),'_','')"; p2_expr = "apoc.text.replace(toLower(r2.pred),'_','')"
    else:
        def heavy(arg: str) -> str:
            return (
                f"apoc.text.replace(apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower({arg}),'freebase.',''),'_',''),'.',''),'/','')"
            )
        w1a_expr = heavy("$w1a"); w2a_expr = heavy("$w2a")
        w1b_expr = heavy("$w1b"); w2b_expr = heavy("$w2b")
        w1c_expr = heavy("$w1c"); w2c_expr = heavy("$w2c")
        p1_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower(r1.pred),'_',''),'.',''),'/','')"
        p2_expr = "apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower(r2.pred),'_',''),'.',''),'/','')"

    qry = (
        f"WITH {w1a_expr} AS w1a, {w2a_expr} AS w2a, {w1b_expr} AS w1b, {w2b_expr} AS w2b, {w1c_expr} AS w1c, {w2c_expr} AS w2c\n"
        "MATCH (s:Resource {id: $startId})-[r1:REL]-(m:Resource)-[r2:REL]-(o:Resource)\n"
        f"WITH s, r1, m, r2, o, {p1_expr} AS np1, {p2_expr} AS np2, w1a, w2a, w1b, w2b, w1c, w2c\n"
        "WITH s, r1, m, r2, o,\n"
        "  CASE WHEN $hasA THEN apoc.text.distance(np1, w1a) ELSE NULL END AS a1,\n"
        "  CASE WHEN $hasA THEN apoc.text.distance(np2, w2a) ELSE NULL END AS a2,\n"
        "  CASE WHEN $hasB THEN apoc.text.distance(np1, w1b) ELSE NULL END AS b1,\n"
        "  CASE WHEN $hasB THEN apoc.text.distance(np2, w2b) ELSE NULL END AS b2,\n"
        "  CASE WHEN $hasC THEN apoc.text.distance(np1, w1c) ELSE NULL END AS c1,\n"
        "  CASE WHEN $hasC THEN apoc.text.distance(np2, w2c) ELSE NULL END AS c2\n"
        "WITH s, r1, m, r2, o, a1, a2, b1, b2, c1, c2,\n"
        "  CASE WHEN a1 IS NULL OR a2 IS NULL THEN NULL ELSE a1 + a2 END AS scoreA,\n"
        "  CASE WHEN b1 IS NULL OR b2 IS NULL THEN NULL ELSE b1 + b2 END AS scoreB,\n"
        "  CASE WHEN c1 IS NULL OR c2 IS NULL THEN NULL ELSE c1 + c2 END AS scoreC\n"
        "WITH s, r1, m, r2, o, scoreA, scoreB, scoreC, [x IN [scoreA, scoreB, scoreC] WHERE x IS NOT NULL] AS scores\n"
        "WHERE (( $hasA AND a1 IS NOT NULL AND a2 IS NOT NULL AND a1 <= $d1 AND a2 <= $d2) OR\n"
        "       ( $hasB AND b1 IS NOT NULL AND b2 IS NOT NULL AND b1 <= $d1 AND b2 <= $d2) OR\n"
        "       ( $hasC AND c1 IS NOT NULL AND c2 IS NOT NULL AND c1 <= $d1 AND c2 <= $d2))\n"
        "WITH s, r1, m, r2, o, scoreA, scoreB, scoreC, scores,\n"
        "  CASE $aggregate WHEN 'sum' THEN reduce(acc=0, x IN scores | acc + x)\n"
        "                  WHEN 'avg' THEN (reduce(acc=0, x IN scores | acc + x) / toFloat(size(scores)))\n"
        "                  ELSE reduce(minVal=1000000000, x IN scores | CASE WHEN x < minVal THEN x ELSE minVal END) END AS score\n"
        "RETURN s.id AS s, r1.pred AS p1, m.id AS m, r2.pred AS p2, o.id AS o, scoreA, scoreB, scoreC, score\n"
        "ORDER BY score ASC\n"
    )
    if limit and limit > 0:
        qry = qry + "LIMIT $limit"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        recs, _, _ = driver.execute_query(
            qry,
            {
                "startId": start_id,
                "w1a": w1a or "", "w2a": w2a or "",
                "w1b": w1b or "", "w2b": w2b or "",
                "w1c": w1c or "", "w2c": w2c or "",
                "hasA": hasA, "hasB": hasB, "hasC": hasC,
                "d1": d1_max, "d2": d2_max, "limit": limit,
                "aggregate": aggregate,
            },
            database_=db,
        )
        return recs
    finally:
        driver.close()


def get_llm(args):
    if args.provider == "openai":
        return OpenAILLM(model_name=args.model)
    return VLLMLLM(model_name=args.model, model_params=None, base_url=args.base_url, api_key=args.api_key)


def build_llama_inst_prompt(question: str, evidence: List[dict]) -> str:
    sys_txt = (
        "Based on the reasoning paths, please answer the given question. "
        "Please keep the answer as simple as possible and return all the possible answers as a list. "
        "Let's think it step by step. Please explain your answer. "
        "Please return each answer in a new line."
    )
    lines = []
    for r in evidence:
        s = r.get("s"); p1 = r.get("p1"); m = r.get("m"); p2 = r.get("p2"); o = r.get("o"); sc = r.get("score")
        lines.append(f"- {s} -[{p1}]-> {m} -[{p2}]-> {o} (score={sc})")
    ev_block = "\n".join(lines) if lines else "(no evidence)"
    return (
        f"[INST] <<SYS>> {sys_txt} <</SYS>>\n\nReasoning Paths:\n{ev_block}\n\nQuestion: {question} [/INST]"
    )


def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    llm = get_llm(args)
    done = 0
    with inp.open("r", encoding="utf-8") as f_in, outp.open("w", encoding="utf-8") as f_out:
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

            pairs: List[Tuple[str, str]] = []
            for idx in (1, 2, 3):
                path = row.get(f"path{idx}") or ""
                sp = split_path(path)
                if sp:
                    pairs.append(sp)
            if not pairs:
                result = {"id": rid, "question": question, "q_entity": q_entity, "paths": [], "answer": "", "no_evidence": True}
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                done += 1
                continue

            try:
                evidence = list(
                    run_query_merged(
                        args.uri, args.user, args.password, args.database,
                        str(q_entity), pairs, args.d1, args.d2, args.limit, args.normalize, args.aggregate
                    )
                )
            except Exception as e:
                result = {"id": rid, "question": question, "q_entity": q_entity, "paths": pairs, "error": str(e)}
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                done += 1
                continue

            if not evidence:
                result = {"id": rid, "question": question, "q_entity": q_entity, "paths": pairs, "answer": "", "no_evidence": True}
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                done += 1
                continue

            top = evidence[: args.topk]
            prompt = build_llama_inst_prompt(question, top)
            try:
                resp = llm.invoke(prompt)
                answer = resp.content.strip()
            except Exception as e:
                answer = f"[llm_error] {e}"

            result = {"id": rid, "question": question, "q_entity": q_entity, "paths": pairs, "answer": answer}
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            done += 1

    print(f"[ok] Wrote {done} rows to {outp}")


if __name__ == "__main__":
    main()
