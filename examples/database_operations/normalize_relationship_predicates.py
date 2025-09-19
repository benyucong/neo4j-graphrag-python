"""Normalize similar relationship predicates using RapidFuzz clustering.

This script clusters similar `r.pred` values on `:REL` relationships and rewrites them
into a canonical predicate, then deduplicates duplicate edges.

Usage:
    python examples/database_operations/normalize_relationship_predicates.py \
        --uri neo4j://localhost:7687 \
        --user neo4j \
        --password password \
        --database neo4j \
        --threshold 0.85 \
        --min-count 1

Install extras if needed:
    pip install "neo4j-graphrag[fuzzy-matching]"
"""

from __future__ import annotations

import argparse
from typing import Optional

from neo4j import GraphDatabase

from neo4j_graphrag.experimental.components.resolver import (
    RelationshipPredicateFuzzyResolver,
)


def parse_args():
    p = argparse.ArgumentParser(description="Normalize similar relationship predicates")
    p.add_argument("--uri", default="neo4j://localhost:7687")
    p.add_argument("--user", default="neo4j")
    p.add_argument("--password", default="password")
    p.add_argument("--database", default="neo4j")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--min-count", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        resolver = RelationshipPredicateFuzzyResolver(
            driver=driver,
            similarity_threshold=args.threshold,
            min_count=args.min_count,
            neo4j_database=args.database,
        )
        stats = asyncio.run(resolver.run())
        print(
            f"Done. Distinct predicates: {stats.number_of_nodes_to_resolve}, clusters: {stats.number_of_created_nodes}"
        )
    finally:
        driver.close()


if __name__ == "__main__":
    import asyncio

    main()
