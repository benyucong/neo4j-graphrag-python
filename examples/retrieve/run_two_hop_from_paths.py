from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

from neo4j import GraphDatabase


def parse_args():
    p = argparse.ArgumentParser(description="Run fuzzy 2-hop Cypher from paths in JSONL.")
    p.add_argument("--input", default="datasets/vanilla_paths_joined.jsonl")
    p.add_argument("--uri", default="neo4j://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--limit", type=int, default=25, help="Max rows per path; 0 means no limit")
    p.add_argument("--d1", type=int, default=3, help="Max Levenshtein distance for first hop")
    p.add_argument("--d2", type=int, default=3, help="Max Levenshtein distance for second hop")
    p.add_argument("--normalize", choices=["none", "simple", "heavy"], default="simple",
                   help="Predicate normalization: none=lowercase; simple=lowercase+remove '_' ; heavy=strip 'freebase.' and remove '_.//'.")
    p.add_argument("--debug-distances", action="store_true", help="Print per-hop distances d1/d2 for each result")
    return p.parse_args()


def split_path(path: str) -> Optional[Tuple[str, str]]:
    if not path:
        return None
    parts = [x.strip() for x in path.split(",")]
    parts = [x for x in parts if x]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def run_query(uri: str, user: str, password: str, db: str, start_id: str, wanted1: str, wanted2: str, d1_max: int, d2_max: int, limit: int, normalize: str) -> Iterable[dict]:
    # Build normalization expressions per chosen mode
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
        qry = qry + "LIMIT $limit"
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        recs, _, _ = driver.execute_query(
            qry,
            {"startId": start_id, "wanted1": wanted1, "wanted2": wanted2, "d1": d1_max, "d2": d2_max, "limit": limit},
            database_=db,
        )
        return recs
    finally:
        driver.close()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = row.get("id")
            q_entity = row.get("q_entity") or row.get("q_entity_id") or row.get("startId")
            if not q_entity:
                print(f"[skip] id={rid} missing q_entity")
                continue

            print(f"\n== id={rid} q_entity={q_entity} ==")
            for idx in (1, 2, 3):
                key = f"path{idx}"
                path = row.get(key) or ""
                hops = split_path(path)
                if not hops:
                    print(f"{key}: [empty or <2 hops]")
                    continue
                p1, p2 = hops
                print(f"{key}: {p1}, {p2}")
                try:
                    recs = run_query(
                        args.uri,
                        args.user,
                        args.password,
                        args.database,
                        str(q_entity),
                        p1,
                        p2,
                        args.d1,
                        args.d2,
                        args.limit,
                        args.normalize,
                    )
                except Exception as e:
                    print(f"[error] query failed: {e}")
                    continue
                recs = list(recs)
                if not recs:
                    print("  No 2-hop matches. First-hop predicate candidates (sample):")
                    # Fallback: show first-hop predicate candidates to help debug
                    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
                    try:
                        q2 = (
                            "MATCH (s:Resource {id: $startId})-[r1:REL]-(m:Resource)\n"
                            "RETURN DISTINCT r1.pred AS pred LIMIT 20"
                        )
                        rows, _, _ = driver.execute_query(q2, {"startId": str(q_entity)}, database_=args.database)
                        for row2 in rows:
                            print(f"    - {row2['pred']}")
                    except Exception as e2:
                        print(f"    [debug failed] {e2}")
                    finally:
                        driver.close()
                    continue
                cnt = 0
                for r in recs:
                    cnt += 1
                    if args.debug_distances:
                        print(
                            f"  {r['s']} -[{r['p1']}]-> {r['m']} -[{r['p2']}]-> {r['o']} (d1={r['d1']}, d2={r['d2']}, score={r['score']})"
                        )
                    else:
                        print(f"  {r['s']} -[{r['p1']}]-> {r['m']} -[{r['p2']}]-> {r['o']} (score={r['score']})")
                print(f"  [total results: {cnt}]")


if __name__ == "__main__":
    main()
