"""End-to-end QA over a generic triples graph using Text2Cypher + GraphRAG.

Graph model assumed (from the generic loader):
  (:Resource {id, datatype?})-[:REL {pred}]->(:Resource)

LLM backends:
  - VLLMLLM (default): requires a running vLLM OpenAI-compatible server
    e.g. python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --host 0.0.0.0 --port 8000
  - OpenAILLM (optional): set OPENAI_API_KEY and switch the llm construction below
"""

from __future__ import annotations

import logging

import neo4j
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm.vllm import VLLMLLM
# from neo4j_graphrag.llm import OpenAILLM  # Uncomment to use OpenAI instead


# Neo4j connection
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
DATABASE = "neo4j"


# Keep schema and examples concise but explicit for the generic model
NEO4J_SCHEMA = """
Node types:
Resource { id: STRING, datatype: STRING (optional) }

Relationship:
REL { pred: STRING }

Patterns:
(:Resource)-[:REL]->(:Resource)

Querying examples:
- Find relation by predicate:
  MATCH (s:Resource {id:'Kismet'})-[:REL {pred:'directed_by'}]->(o:Resource) RETURN o.id
- List tags:
  MATCH (:Resource {id:'Kismet'})-[:REL {pred:'has_tags'}]->(t:Resource) RETURN t.id
- Release year:
  MATCH (:Resource {id:'Kismet'})-[:REL {pred:'release_year'}]->(y:Resource) RETURN y.id
"""

EXAMPLES = [
    "USER INPUT: 'Who directed Kismet?' QUERY: MATCH (:Resource {id:'Kismet'})-[:REL {pred:'directed_by'}]->(p:Resource) RETURN p.id",
    "USER INPUT: 'Which actors starred in Kismet?' QUERY: MATCH (:Resource {id:'Kismet'})-[:REL {pred:'starred_actors'}]->(a:Resource) RETURN a.id",
    "USER INPUT: 'What is the release year of Kismet?' QUERY: MATCH (:Resource {id:'Kismet'})-[:REL {pred:'release_year'}]->(y:Resource) RETURN y.id",
]


def main() -> None:
    logging.basicConfig(format="%(asctime)s - %(message)s")
    logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)

    # Choose LLM backend.
    # Default: vLLM server via OpenAI-compatible API
    llm = VLLMLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        model_params={"temperature": 0},
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    # Or: OpenAI
    # llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema=NEO4J_SCHEMA,
            examples=EXAMPLES,
            neo4j_database=DATABASE,
        )

        rag = GraphRAG(retriever=retriever, llm=llm)

        question = "Who directed Kismet?"
        result = rag.search(
            query_text=question,
            return_context=True,
            response_fallback="I can't answer from the graph.",
        )

    print("Q:", question)
    print("A:", result.answer)
    # print(result.retriever_result)  # Uncomment to inspect retrieval context


if __name__ == "__main__":
    main()
