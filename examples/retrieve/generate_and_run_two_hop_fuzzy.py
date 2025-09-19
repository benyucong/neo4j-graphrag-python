"""LLM infers wanted1/wanted2 from question, then executes fuzzy 2-hop query.

Flow:
 1) Ask LLM to output JSON {wanted1,wanted2,sHint,mHint,oHint}
 2) Run APOC Levenshtein 2-hop query with those as parameters

Usage:
  python examples/retrieve/generate_and_run_two_hop_fuzzy.py \
    --question "Show movies directed by Hitchcock and starring Cary Grant" \
    --d1 3 --d2 3 --limit 25 \
    --uri neo4j://localhost:7687 --user neo4j --password password --database neo4j
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM, VLLMLLM


EXTRACT_PROMPT_PATH = Path(__file__).parent / "prompts" / "two_hop_predicate_extraction_system.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Infer and run fuzzy 2-hop query")
    p.add_argument("--question", required=True)
    p.add_argument("--d1", type=int, default=3)
    p.add_argument("--d2", type=int, default=3)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--uri", default="neo4j://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--provider", choices=["openai","vllm"], default="vllm")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="sk-noop")
    return p.parse_args()


def get_llm(args):
    if args.provider == "openai":
        return OpenAILLM(model_name=args.model)
    return VLLMLLM(
        model_name=args.model,
        model_params={"base_url": args.base_url, "api_key": args.api_key},
    )


def infer_predicates(llm, question: str):
    system = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    res = llm.invoke(question, system_instruction=system)
    text = res.content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON: {text}") from e
    for key in ["wanted1","wanted2","startId","sExact","sHint","mHint","oHint"]:
        data.setdefault(key, None)
    return data


def run_query(uri: str, user: str, password: str, db: str, params: dict):
        qry = """
        WITH toLower($wanted1) AS w1, toLower($wanted2) AS w2
        CALL {
            WITH $startId AS startId, $sExact AS sExact
            WITH startId, sExact
            // If startId provided, anchor the start node; else match all
            CALL {
                WITH startId, sExact
                WITH startId, sExact
                // exact match when sExact is true
                CALL {
                    WITH startId, sExact
                    WITH startId, sExact
                    MATCH (s:Resource)
                    WHERE startId IS NOT NULL AND sExact = true AND s.id = startId
                    RETURN collect(s) AS nodes
                    UNION
                    WITH startId, sExact
                    MATCH (s:Resource)
                    WHERE startId IS NOT NULL AND (sExact = false OR sExact IS NULL) AND toLower(s.id) CONTAINS toLower(startId)
                    RETURN collect(s) AS nodes
                    UNION
                    WITH startId, sExact
                    MATCH (s:Resource)
                    WHERE startId IS NULL
                    RETURN collect(s) AS nodes
                }
                RETURN nodes
            }
            UNWIND nodes AS s
            RETURN collect(s) AS starts
        }
        UNWIND starts AS s
        MATCH (s)-[r1:REL]->(m:Resource)-[r2:REL]->(o:Resource)
        WITH s, r1, m, r2, o,
                 apoc.text.levenshteinDistance(
                     apoc.text.replace(toLower(r1.pred),'_',''), apoc.text.replace(w1,'_','')
                 ) AS d1,
                 apoc.text.levenshteinDistance(
                     apoc.text.replace(toLower(r2.pred),'_',''), apoc.text.replace(w2,'_','')
                 ) AS d2
        WHERE d1 <= $d1 AND d2 <= $d2
            AND ($sHint IS NULL OR toLower(s.id) CONTAINS toLower($sHint))
            AND ($mHint IS NULL OR toLower(m.id) CONTAINS toLower($mHint))
            AND ($oHint IS NULL OR toLower(o.id) CONTAINS toLower($oHint))
        RETURN s.id AS s, r1.pred AS p1, m.id AS m, r2.pred AS p2, o.id AS o, (d1 + d2) AS score
        ORDER BY score ASC
        LIMIT $limit
        """
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
                recs, _, _ = driver.execute_query(qry, params, database_=db)
                return recs
        finally:
                driver.close()


def main():
    args = parse_args()
    llm = get_llm(args)
    data = infer_predicates(llm, args.question)
    params = {
        "wanted1": data.get("wanted1"),
        "wanted2": data.get("wanted2"),
        "startId": data.get("startId"),
        "sExact": bool(data.get("sExact")) if data.get("sExact") is not None else None,
        "sHint": data.get("sHint") or None,
        "mHint": data.get("mHint") or None,
        "oHint": data.get("oHint") or None,
        "d1": args.d1,
        "d2": args.d2,
        "limit": args.limit,
    }
    print("Inferred:", params)
    rows = run_query(args.uri, args.user, args.password, args.database, params)
    for r in rows:
        print(f"{r['s']} -[{r['p1']}]-> {r['m']} -[{r['p2']}]-> {r['o']} (score={r['score']})")


if __name__ == "__main__":
    main()
