"""Two-hop direct fuzzy scoring example using APOC Levenshtein.

This script matches two-hop patterns while fuzzy-scoring relationship predicates
inline (no data mutation, no app-side candidate computation).

Usage:
    python examples/retrieve/fuzzy_predicate_two_hop_direct.py \
        --wanted1 director_of \
        --wanted2 written_by \
        --d1 3 --d2 3 \
        --uri neo4j://localhost:7687 \
        --user neo4j \
        --password password \
        --database neo4j

Requires APOC installed on the Neo4j server.
"""

from __future__ import annotations

import argparse
from neo4j import GraphDatabase


def parse_args():
    p = argparse.ArgumentParser(description="Two-hop direct fuzzy scoring")
    p.add_argument("--wanted1", required=True, help="First hop desired predicate")
    p.add_argument("--wanted2", required=True, help="Second hop desired predicate")
    p.add_argument("--d1", type=int, default=3, help="Max Levenshtein distance for hop 1")
    p.add_argument("--d2", type=int, default=3, help="Max Levenshtein distance for hop 2")
    p.add_argument("--uri", default="neo4j://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--limit", type=int, default=50)
    return p.parse_args()


def run_query(driver, wanted1: str, wanted2: str, d1: int, d2: int, db: str, limit: int):
    qry = f"""
    WITH toLower($w1) AS w1, toLower($w2) AS w2
    MATCH (s:Resource)-[r1:REL]->(m:Resource)-[r2:REL]->(o:Resource)
    WITH s, r1, m, r2, o,
         apoc.text.levenshteinDistance(
           apoc.text.replace(toLower(r1.pred),'_',''), apoc.text.replace(w1,'_','')
         ) AS d1,
         apoc.text.levenshteinDistance(
           apoc.text.replace(toLower(r2.pred),'_',''), apoc.text.replace(w2,'_','')
         ) AS d2
    WHERE d1 <= $d1 AND d2 <= $d2
    RETURN s.id AS s, r1.pred AS p1, m.id AS m, r2.pred AS p2, o.id AS o, (d1 + d2) AS score
    ORDER BY score ASC
    LIMIT $limit
    """
    recs, _, _ = driver.execute_query(
        qry,
        {"w1": wanted1, "w2": wanted2, "d1": d1, "d2": d2, "limit": limit},
        database_=db,
    )
    return recs


def main():
    args = parse_args()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        rows = run_query(
            driver,
            args.wanted1,
            args.wanted2,
            args.d1,
            args.d2,
            args.database,
            args.limit,
        )
        for r in rows:
            print(f"{r['s']} -[{r['p1']}]-> {r['m']} -[{r['p2']}]-> {r['o']} (score={r['score']})")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
