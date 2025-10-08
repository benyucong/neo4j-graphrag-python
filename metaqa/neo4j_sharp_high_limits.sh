#!/usr/bin/env bash
# Sweep higher --limit values for merged path batch answering.
# Limits tested: 16 20 25 30 50 100

#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --mem=120G
#SBATCH --gres=gpu:1

set -x
IFS=$'\n\t'

ml cuda gcc mamba
conda activate neo4j

# ----------------- Config -----------------
NEO4J_HOME=${NEO4J_HOME:-/scratch/cs/adis/yuc10/neo4j-community-5.26.0}
WORKDIR=${WORKDIR:-/scratch/cs/adis/yuc10/neo4j-graphrag-python}
MODEL=${MODEL:-Qwen/Qwen3-4B}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_WAIT_TRIES=${VLLM_WAIT_TRIES:-500}
OPENAI_API_BASE=${OPENAI_API_BASE:-http://localhost:${VLLM_PORT}/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-sk-noop}

if [[ "$MODEL" == meta-llama/* ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[warn] HUGGING_FACE_HUB_TOKEN not set for gated model $MODEL" >&2
fi

NEO4J_URI=${NEO4J_URI:-neo4j://localhost:7687}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}
NEO4J_DATABASE=${NEO4J_DATABASE:-neo4j}

cleanup() {
  echo "[cleanup] stopping services..." >&2
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  if [[ "${NEO4J_STARTED_BY_THIS:-0}" -eq 1 ]]; then
    if [[ -x "$NEO4J_HOME/bin/neo4j" ]]; then
      "$NEO4J_HOME/bin/neo4j" stop || true
    else
      "$NEO4J_HOME/bin/neo4j-admin" server stop || true
    fi
  fi
}
trap cleanup EXIT INT TERM

echo "[info] Ensuring Neo4j is running from $NEO4J_HOME" >&2
cd "$NEO4J_HOME" || { echo "[error] NEO4J_HOME '$NEO4J_HOME' not found" >&2; exit 1; }
if "$NEO4J_HOME/bin/cypher-shell" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
  echo "[info] Neo4j already running" >&2; NEO4J_STARTED_BY_THIS=0
else
  echo "[info] Starting Neo4j..." >&2
  if [[ -x bin/neo4j ]]; then bin/neo4j start; else bin/neo4j-admin server start; fi
  NEO4J_STARTED_BY_THIS=1
fi

echo "[info] Waiting for Neo4j to accept connections..." >&2
for i in {1..60}; do
  if "$NEO4J_HOME/bin/cypher-shell" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
    echo "[ok] Neo4j is up" >&2; break
  fi
  sleep 2
  if [[ "$i" -eq 60 ]]; then echo "[error] Neo4j did not become ready" >&2; exit 1; fi
done

echo "[info] Activating Python environment and launching vLLM" >&2
cd "$WORKDIR" || { echo "[error] WORKDIR '$WORKDIR' not found" >&2; exit 1; }
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

mkdir -p logs_sharp_high
VLLM_CMD=(python -m vllm.entrypoints.openai.api_server --model "$MODEL" --host "$VLLM_HOST" --port "$VLLM_PORT" --dtype auto --download-dir "$HF_HOME")
if [[ -n "$VLLM_EXTRA_ARGS" ]]; then VLLM_CMD+=($VLLM_EXTRA_ARGS); fi
"${VLLM_CMD[@]}" > logs_sharp_high/vllm.out 2> logs_sharp_high/vllm.err &
VLLM_PID=$!
echo "[info] vLLM started with PID $VLLM_PID" >&2

echo "[info] Waiting for vLLM OpenAI API at $OPENAI_API_BASE ..." >&2
tries=0
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[error] vLLM exited early" >&2
    tail -n 80 logs_sharp_high/vllm.err >&2 || true
    tail -n 40 logs_sharp_high/vllm.out >&2 || true
    exit 1
  fi
  if curl -sf "$OPENAI_API_BASE/models" >/dev/null 2>&1; then
    echo "[ok] vLLM API is up" >&2; break
  fi
  tries=$((tries+1))
  if [[ "$tries" -ge "$VLLM_WAIT_TRIES" ]]; then
    echo "[error] vLLM not ready in time" >&2
    tail -n 80 logs_sharp_high/vllm.err >&2 || true
    tail -n 40 logs_sharp_high/vllm.out >&2 || true
    exit 1
  fi
  sleep 2
done

echo "[run] Generating answers for limits: 16 20 25 30 50 100" >&2
DATASET_PATH=${DATASET_PATH:-datasets/vanilla_paths_joined.jsonl}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-outputs}
mkdir -p "$BASE_OUTPUT_DIR"

LIMITS=(16 20 25 30 50 100)

for L in "${LIMITS[@]}"; do
  # scale d1/d2 (more generous as limit grows); clamp d2 at 80
  if   (( L <= 20 )); then D1=40; D2=60; fi
  if   (( L > 20 && L <= 30 )); then D1=50; D2=70; fi
  if   (( L > 30 && L <= 50 )); then D1=60; D2=80; fi
  if   (( L > 50 )); then D1=70; D2=80; fi

  OUTPUT_PATH="${BASE_OUTPUT_DIR}/answers_merged_sharp_limit${L}.jsonl"
  echo "[run] limit=${L}, d1=${D1}, d2=${D2} -> $OUTPUT_PATH" >&2

  python examples/retrieve/batch_answers_from_dataset.py \
    --input "$DATASET_PATH" \
    --output "$OUTPUT_PATH" \
    --aggregate min \
    --d1 "$D1" \
    --d2 "$D2" \
    --limit "$L" \
    --normalize none \
    --topk "$L" \
    --max-rows "${BATCH_MAX_ROWS:-0}" \
    --uri "$NEO4J_URI" --user "$NEO4J_USER" --password "$NEO4J_PASSWORD" --database "$NEO4J_DATABASE" \
    --provider vllm --model "$MODEL" --base-url "$OPENAI_API_BASE" --api-key "$OPENAI_API_KEY" \
    || { echo "[error] Batch answering failed for limit=$L" >&2; exit 1; }

  echo "[ok] Answers written to $OUTPUT_PATH" >&2
  tail -n 3 "$OUTPUT_PATH" || true
  METRICS_PATH="${OUTPUT_PATH}.metrics.json"
  if [[ -f "$METRICS_PATH" ]]; then
    echo "[metrics] (limit=$L):" >&2
    jq '{rows_processed, elapsed_seconds, throughput_rows_per_sec}' "$METRICS_PATH" 2>/dev/null || cat "$METRICS_PATH" || true
  fi
done

echo "[done] High-limit sweep complete" >&2
