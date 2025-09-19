"""Query-time fuzzy predicate matching example using RapidFuzz.

This script demonstrates selecting top-K similar relationship predicates (stored on
`:REL {pred}`) to a desired term (e.g., "director_of"), then querying using equality
against those candidates for performance.

Usage:
    python examples/retrieve/fuzzy_predicate_retriever.py \
        --wanted director_of \
        --k 5 \
        --uri neo4j://localhost:7687 \
        --user neo4j \
        --password password \
        --database neo4j

Requires:
    pip install rapidfuzz
"""

from __future__ import annotations

import argparse
from typing import List

from neo4j import GraphDatabase
from rapidfuzz import fuzz, utils


def parse_args():
    p = argparse.ArgumentParser(description="Query-time fuzzy predicate matcher")
    p.add_argument("--wanted", required=True, help="Desired predicate, e.g., director_of")
    p.add_argument("--k", type=int, default=5, help="Top-K candidates to use")
    p.add_argument("--uri", default="neo4j://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    return p.parse_args()


def get_predicate_candidates(driver, wanted: str, k: int, db: str) -> List[str]:
    qry = """
    MATCH ()-[r:REL]->()
    WITH toLower(r.pred) AS p, count(*) AS c
    RETURN p AS pred, c AS cnt
    """
    recs, _, _ = driver.execute_query(qry, database_=db)
    items = [(r["pred"], r["cnt"]) for r in recs if r["pred"]]

    # score by similarity (RapidFuzz WRatio) after default normalization
    scored = [
        (p, fuzz.WRatio(p, wanted.lower(), processor=utils.default_process))
        for p, _ in items
    ]
    scored.sort(key=lambda x: -x[1])
    return [p for p, _ in scored[: max(k, 1)]]


def run_query(driver, candidates: List[str], db: str):
    qry = """
    MATCH (s:Resource)-[r:REL]->(o:Resource)
    WHERE toLower(r.pred) IN $preds
    RETURN s.id AS s, r.pred AS pred, o.id AS o
    LIMIT 50
    """
    recs, _, _ = driver.execute_query(qry, {"preds": candidates}, database_=db)
    return recs


def main():
    args = parse_args()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        preds = get_predicate_candidates(driver, args.wanted, args.k, args.database)
        print(f"Candidates for '{args.wanted}': {preds}")
        rows = run_query(driver, preds, args.database)
        for r in rows:
            print(f"{r['s']} -[{r['pred']}]-> {r['o']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
