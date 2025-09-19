"""Load a simple pipe-delimited triples text file into Neo4j (generic model).

Each line should be:

    Subject|predicate|Object

Modeling:
    - All nodes (subjects and objects) become `(:Resource {id})`
    - For objects, we also annotate a `datatype` property: one of {"integer","float","boolean","string"}
    - Every triple is an edge `(:Resource)-[:REL {pred}]->(:Resource)` with the original predicate stored as a relationship property

This keeps ingestion schema-agnostic and uniform:
    MATCH (s:Resource {id: "Kismet"})-[:REL {pred: "directed_by"}]->(o:Resource)

Usage:
    python examples/build_graph/load_triples_from_text.py \
        --file /path/to/triples.txt \
        --uri neo4j://localhost:7687 \
        --user neo4j \
        --password password \
        --database neo4j
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import neo4j


@dataclass
class Config:
    file: Path
    uri: str
    user: str
    password: str
    database: Optional[str]


def _parse_literal(raw: str) -> Tuple[bool, object, str]:
    """Return (is_literal, value, datatype)."""
    s = raw.strip()
    # boolean
    if s.lower() in {"true", "false"}:
        return True, s.lower() == "true", "boolean"
    # integer
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        try:
            return True, int(s), "integer"
        except ValueError:
            pass
    # float
    try:
        if "." in s or "e" in s.lower():
            f = float(s)
            return True, f, "float"
    except ValueError:
        pass
    # default string
    return True, s, "string"


def load_line(session: neo4j.Session, subject: str, predicate: str, obj: str) -> None:
    pred = predicate.strip()
    # subject always a Resource
    session.run("MERGE (s:Resource {id: $id})", id=subject)

    # object also a Resource; its id is the string representation, with datatype annotation
    is_lit, value, dtype = _parse_literal(obj)
    obj_id = str(value)
    session.run(
        """
        MERGE (s:Resource {id: $sid})
        MERGE (o:Resource {id: $oid})
        ON CREATE SET o.datatype = $dtype
        ON MATCH SET o.datatype = coalesce(o.datatype, $dtype)
        MERGE (s)-[:REL {pred: $pred}]->(o)
        """,
        sid=subject,
        oid=obj_id,
        dtype=dtype,
        pred=pred,
    )


def run(cfg: Config) -> None:
    driver = neo4j.GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    try:
        with driver.session(database=cfg.database) as session:
            # Optional: minimal constraints for generic model
            session.run(
                "CREATE CONSTRAINT resource_id IF NOT EXISTS FOR (r:Resource) REQUIRE r.id IS UNIQUE"
            )

            with cfg.file.open("r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, start=1):
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("|") 
                    if len(parts) < 3:
                        print(f"Skipping malformed line {lineno}: {line}", file=sys.stderr)
                        continue
                    subj, pred, obj = parts[0].strip(), parts[1].strip(), "|".join(parts[2:]).strip()
                    load_line(session, subj, pred, obj)
    finally:
        driver.close()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Load triples text into Neo4j")
    parser.add_argument("--file", required=True, type=Path, help="Path to triples file")
    parser.add_argument("--uri", default="neo4j://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    args = parser.parse_args()
    return Config(file=args.file, uri=args.uri, user=args.user, password=args.password, database=args.database)


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
