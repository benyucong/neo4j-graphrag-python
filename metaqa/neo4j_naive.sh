#!/usr/bin/env bash
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu-h100-80g
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"

set -x
IFS=$'\n\t'

ml cuda gcc mamba

conda activate neo4j

# Configuration (override via env vars if desired)
NEO4J_HOME=${NEO4J_HOME:-/scratch/cs/adis/yuc10/neo4j-community-5.26.0}
WORKDIR=${WORKDIR:-/scratch/cs/adis/yuc10/neo4j-graphrag-python}
# MODEL=${MODEL:-rmanluo/RoG}
MODEL=${MODEL:-Qwen/Qwen3-4B}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8001}
# How many attempts (2s each) to wait for vLLM readiness
VLLM_WAIT_TRIES=${VLLM_WAIT_TRIES:-500}
OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:${VLLM_PORT}/v1}
# The OpenAI SDK needs a key even if server auth is disabled
export OPENAI_API_KEY=${OPENAI_API_KEY:-sk-noop}

# Warn if trying to use gated HF models without a token (e.g., Llama 2)
if [[ "$MODEL" == meta-llama/* ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[warn] HUGGING_FACE_HUB_TOKEN not set. Gated models like '$MODEL' require Hugging Face access." >&2
  echo "       Accept the license on the model page and set: export HUGGING_FACE_HUB_TOKEN=..." >&2
fi

NEO4J_URI=${NEO4J_URI:-neo4j://localhost:7687}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}
NEO4J_DATABASE=${NEO4J_DATABASE:-neo4j}

cleanup() {
  echo "[cleanup] stopping services..." >&2
  # Stop vLLM server if running
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[cleanup] killing vLLM PID $VLLM_PID" >&2
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  # Stop Neo4j
  if [[ "${NEO4J_STARTED_BY_THIS:-0}" -eq 1 ]]; then
    if [[ -x "$NEO4J_HOME/bin/neo4j" ]]; then
      "$NEO4J_HOME/bin/neo4j" stop || true
    else
      "$NEO4J_HOME/bin/neo4j-admin" server stop || true
    fi
  else
    echo "[cleanup] leaving Neo4j running (not started by this job)" >&2
  fi
}
trap cleanup EXIT INT TERM

echo "[info] Ensuring Neo4j is running from $NEO4J_HOME" >&2
cd "$NEO4J_HOME" || { echo "[error] NEO4J_HOME '$NEO4J_HOME' not found" >&2; exit 1; }
if "$NEO4J_HOME/bin/cypher-shell" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
  echo "[info] Neo4j already running" >&2
  NEO4J_STARTED_BY_THIS=0
else
  echo "[info] Starting Neo4j..." >&2
  if [[ -x bin/neo4j ]]; then
    bin/neo4j start
  else
    bin/neo4j-admin server start
  fi
  NEO4J_STARTED_BY_THIS=1
fi

# Wait for Neo4j bolt port to be ready
echo "[info] Waiting for Neo4j to accept connections..." >&2
for i in {1..60}; do
  if "$NEO4J_HOME/bin/cypher-shell" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
    echo "[ok] Neo4j is up" >&2
    break
  fi
  sleep 2
  if [[ "$i" -eq 60 ]]; then
    echo "[error] Neo4j did not become ready in time" >&2
    exit 1
  fi
done

echo "[info] Activating Python environment and launching vLLM" >&2
cd "$WORKDIR" || { echo "[error] WORKDIR '$WORKDIR' not found" >&2; exit 1; }

# Prefer local venv if present; otherwise assume environment preconfigured
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

mkdir -p logs_naive
VLLM_CMD=(python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --dtype auto
  --download-dir "$HF_HOME"
)
if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  VLLM_CMD+=($VLLM_EXTRA_ARGS)
fi
"${VLLM_CMD[@]}" > logs_naive/vllm.out 2> logs_naive/vllm.err &
VLLM_PID=$!
echo "[info] vLLM started with PID $VLLM_PID" >&2

echo "[info] Waiting for vLLM OpenAI API at $OPENAI_API_BASE ..." >&2
tries=0
while true; do
  # If process died, abort early and show logs
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
  echo "[error] vLLM process exited early (PID $VLLM_PID). Showing recent logs:" >&2
  echo "--- logs_naive/vllm.err (tail -n 100) ---" >&2
  tail -n 100 logs_naive/vllm.err >&2 || true
  echo "--- logs_naive/vllm.out (tail -n 50) ---" >&2
  tail -n 50 logs_naive/vllm.out >&2 || true
    exit 1
  fi
  if curl -sf "$OPENAI_API_BASE/models" >/dev/null 2>&1; then
    echo "[ok] vLLM API is up" >&2
    break
  fi
  tries=$((tries+1))
  if [[ "$tries" -ge "$VLLM_WAIT_TRIES" ]]; then
  echo "[error] vLLM did not become ready in time. Showing recent logs:" >&2
  echo "--- logs_naive/vllm.err (tail -n 100) ---" >&2
  tail -n 100 logs_naive/vllm.err >&2 || true
  echo "--- logs_naive/vllm.out (tail -n 50) ---" >&2
  tail -n 50 logs_naive/vllm.out >&2 || true
    echo "[hint] If you are using a gated HF model (e.g., Llama 2), ensure HUGGING_FACE_HUB_TOKEN is set." >&2
    echo "[hint] Try an open model, e.g., export MODEL=mistralai/Mistral-7B-Instruct-v0.2" >&2
    exit 1
  fi
  sleep 2
done

echo "[run] Generating answers over dataset via sequential 3-path queries" >&2
DATASET_PATH=${DATASET_PATH:-datasets/vanilla_paths_joined.jsonl}
# DATASET_PATH=${DATASET_PATH:-datasets/vanilla_demo.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/answers_merged_naive.jsonl}
mkdir -p "$(dirname "$OUTPUT_PATH")"
python examples/retrieve/batch_answers_from_dataset_seq.py \
  --input "$DATASET_PATH" \
  --output "$OUTPUT_PATH" \
  --aggregate min --d1 60 --d2 60 --limit 0 --normalize none --topk 1000000 \
  --max-rows "${BATCH_MAX_ROWS:-0}" \
  --uri "$NEO4J_URI" --user "$NEO4J_USER" --password "$NEO4J_PASSWORD" --database "$NEO4J_DATABASE" \
  --provider vllm --model "$MODEL" --base-url "$OPENAI_API_BASE" --api-key "$OPENAI_API_KEY" \
  || { echo "[error] Batch answering failed" >&2; exit 1; }

echo "[ok] Answers written to $OUTPUT_PATH" >&2
echo "[sample] Showing last 5 answers:" >&2
tail -n 5 "$OUTPUT_PATH" || true

# Show throughput metrics if available
METRICS_PATH="${OUTPUT_PATH}.metrics.json"
if [[ -f "$METRICS_PATH" ]]; then
  echo "[metrics] Summary:" >&2
  jq '{rows_processed, elapsed_seconds, throughput_rows_per_sec} // .' "$METRICS_PATH" 2>/dev/null || cat "$METRICS_PATH" || true
fi



