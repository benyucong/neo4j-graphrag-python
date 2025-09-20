from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

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
    p.add_argument("--aggregate", choices=["min", "sum", "avg"], default="min",
                   help="How to combine scores from up to three paths: min (best), sum, or avg of available path scores.")
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
    # Ensure exactly 3 entries by padding with empty strings
    p = pairs[:3] + [("", "")] * (3 - len(pairs))
    (w1a, w2a), (w1b, w2b), (w1c, w2c) = p
    hasA = bool(w1a and w2a)
    hasB = bool(w1b and w2b)
    hasC = bool(w1c and w2c)

    # Build normalization expressions per chosen mode
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
    else:  # heavy
        def heavy(arg: str) -> str:
            return (
                f"apoc.text.replace(apoc.text.replace(apoc.text.replace(apoc.text.replace(toLower({arg}),'freebase.',''),'_',''),'.',''),'/','')"
            )
        w1a_expr = heavy("$w1a"); w2a_expr = heavy("$w2a")
        w1b_expr = heavy("$w1b"); w2b_expr = heavy("$w2b")
        w1c_expr = heavy("$w1c"); w2c_expr = heavy("$w2c")
        # For predicates on relationships, we don't strip 'freebase.' prefix since it's unlikely present on r.pred, but we normalize separators
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
        "WITH s, r1, m, r2, o, a1, a2, b1, b2, c1, c2, scoreA, scoreB, scoreC,\n"
        "  [x IN [scoreA, scoreB, scoreC] WHERE x IS NOT NULL] AS scores\n"
        "WHERE (\n"
        "  ($hasA AND a1 IS NOT NULL AND a2 IS NOT NULL AND a1 <= $d1 AND a2 <= $d2) OR\n"
        "  ($hasB AND b1 IS NOT NULL AND b2 IS NOT NULL AND b1 <= $d1 AND b2 <= $d2) OR\n"
        "  ($hasC AND c1 IS NOT NULL AND c2 IS NOT NULL AND c1 <= $d1 AND c2 <= $d2)\n"
        ")\n"
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

            # Gather up to three path pairs
            pairs: List[Tuple[str, str]] = []
            for idx in (1, 2, 3):
                key = f"path{idx}"
                hops = split_path(row.get(key) or "")
                if hops:
                    pairs.append(hops)

            if not pairs:
                print(f"\n== id={rid} q_entity={q_entity} ==\n[skip] no valid paths in row")
                continue

            print(f"\n== id={rid} q_entity={q_entity} ==")
            for i, (pp1, pp2) in enumerate(pairs, start=1):
                print(f"path{i}: {pp1}, {pp2}")
            try:
                recs = run_query_merged(
                    args.uri,
                    args.user,
                    args.password,
                    args.database,
                    str(q_entity),
                    pairs,
                    args.d1,
                    args.d2,
                    args.limit,
                    args.normalize,
                    args.aggregate,
                )
            except Exception as e:
                print(f"[error] query failed: {e}")
                continue
            recs = list(recs)
            if not recs:
                print("  No 2-hop matches across provided paths. First-hop predicate candidates (sample):")
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
                    def fmt(x):
                        return "N/A" if x is None else str(x)
                    print(
                        f"  {r['s']} -[{r['p1']}]-> {r['m']} -[{r['p2']}]-> {r['o']} (scoreA={fmt(r.get('scoreA'))}, scoreB={fmt(r.get('scoreB'))}, scoreC={fmt(r.get('scoreC'))}, combined={r['score']})"
                    )
                else:
                    print(f"  {r['s']} -[{r['p1']}]-> {r['m']} -[{r['p2']}]-> {r['o']} (score={r['score']})")
            print(f"  [total results: {cnt}]")


if __name__ == "__main__":
    main()
