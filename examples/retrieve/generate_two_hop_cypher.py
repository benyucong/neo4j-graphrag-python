"""Generate a 2-hop fuzzy Cypher with an LLM and print it + params.

Environment:
 - Choose LLM via code below (OpenAI, VLLM, etc.).
 - Requires an LLM API/key depending on provider.

Usage:
  python examples/retrieve/generate_two_hop_cypher.py \
    --question "Show movies directed by Hitchcock and starring Cary Grant" \
    --wanted1 director_of --wanted2 acted_in \
    --d1 3 --d2 3 --limit 25 \
    --sHint "" --mHint "Hitchcock" --oHint "Cary Grant"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neo4j_graphrag.llm import OpenAILLM, VLLMLLM


def parse_args():
    p = argparse.ArgumentParser(description="Generate 2-hop fuzzy Cypher via LLM")
    p.add_argument("--question", required=True)
    p.add_argument("--wanted1", required=True)
    p.add_argument("--wanted2", required=True)
    p.add_argument("--d1", type=int, default=3)
    p.add_argument("--d2", type=int, default=3)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--sHint", default="")
    p.add_argument("--mHint", default="")
    p.add_argument("--oHint", default="")
    p.add_argument("--provider", choices=["openai","vllm"], default="vllm")
    p.add_argument("--model", default="gpt-4o-mini")
    # vLLM/OpenAI-compatible settings
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="sk-noop")
    # OpenAI cloud: set OPENAI_API_KEY env or override --api-key/--base-url as needed
    return p.parse_args()


def load_system_prompt() -> str:
    path = Path(__file__).parent / "prompts" / "two_hop_fuzzy_apoc_system.txt"
    return path.read_text(encoding="utf-8")


def get_llm(args):
    if args.provider == "openai":
        return OpenAILLM(model_name=args.model)
    # default: vLLM via OpenAI-compatible API
    return VLLMLLM(
        model_name=args.model,
        model_params={"base_url": args.base_url, "api_key": args.api_key},
    )


def main():
    args = parse_args()
    llm = get_llm(args)
    system = load_system_prompt()

    user = f"Question: {args.question}\nGenerate the 2-hop fuzzy Cypher query."
    res = llm.invoke(user, system_instruction=system)
    print("--- Generated Cypher ---")
    print(res.content.strip())
    print("\n--- Suggested Parameters ---")
    params = {
        "wanted1": args.wanted1,
        "wanted2": args.wanted2,
        "d1": args.d1,
        "d2": args.d2,
        "limit": args.limit,
        "sHint": args.sHint or None,
        "mHint": args.mHint or None,
        "oHint": args.oHint or None,
    }
    print(params)


if __name__ == "__main__":
    main()
