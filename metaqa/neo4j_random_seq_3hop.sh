#!/usr/bin/env bash
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu-h200-141g-ellis,gpu-a100-80g,gpu-h100-80g
#SBATCH --mem=100G
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
  echo "[warn] HUGGING_FACE_HUB_TOKEN not set. Gated models like '$MODEL' require HF access." >&2
fi

NEO4J_URI=${NEO4J_URI:-neo4j://localhost:7687}
NEO4J_USER=${NEO4J_USER:-neo4j}
NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}
NEO4J_DATABASE=${NEO4J_DATABASE:-neo4j}

# Sweep config
DATASET_PATH=${DATASET_PATH:-datasets/vanilla_paths_joined_3hop.jsonl}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-random_seq_outputs_3hop}
mkdir -p "$BASE_OUTPUT_DIR"

# -------- Linear budget controls for 3-hop ----------
# Work budget W(L) ~= K * L ; then d1 ~ cbrt(W), d2 ~ d1, d3 ~ d1
# For 3-hop, we distribute budget across 3 hops
K=${K:-64}
# Keep prompt size steady to avoid LLM-time step jumps
TOPK_FIXED=${TOPK_FIXED:-25}
# Optional hard cap on d1/d2/d3 (keeps extremes in check)
D1_MAX=${D1_MAX:-1000}
D2_MAX=${D2_MAX:-1000}
D3_MAX=${D3_MAX:-1000}

# Limits to sweep
ALL_LIMITS=()
for L in $(seq 1 15); do ALL_LIMITS+=("$L"); done
ALL_LIMITS+=(16 20 25 30 50 100)

# ----------------- Cleanup -----------------
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

# ----------------- Neo4j -----------------
echo "[info] Ensuring Neo4j is running from $NEO4J_HOME" >&2
cd "$NEO4J_HOME" || { echo "[error] NEO4J_HOME '$NEO4J_HOME' not found" >&2; exit 1; }
if "$NEO4J_HOME/bin/cypher-shell" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "RETURN 1;" >/dev/null 2>&1; then
  echo "[info] Neo4j already running" >&2
  NEO4J_STARTED_BY_THIS=0
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
  if [[ "$i" -eq 60 ]]; then echo "[error] Neo4j did not become ready in time" >&2; exit 1; fi
done

# ----------------- vLLM -----------------
echo "[info] Activating Python environment and launching vLLM" >&2
cd "$WORKDIR" || { echo "[error] WORKDIR '$WORKDIR' not found" >&2; exit 1; }
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

mkdir -p logs_random_seq_3hop
VLLM_CMD=(python -m vllm.entrypoints.openai.api_server
  --model "$MODEL" --host "$VLLM_HOST" --port "$VLLM_PORT"
  --dtype auto --download-dir "${HF_HOME:-$HOME/.cache/huggingface}"
)
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then VLLM_CMD+=($VLLM_EXTRA_ARGS); fi
"${VLLM_CMD[@]}" > logs_random_seq_3hop/vllm.out 2> logs_random_seq_3hop/vllm.err &
VLLM_PID=$!
echo "[info] vLLM started with PID $VLLM_PID" >&2

echo "[info] Waiting for vLLM OpenAI API at $OPENAI_API_BASE ..." >&2
tries=0
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[error] vLLM exited early. Recent logs:" >&2
    tail -n 100 logs_random_seq_3hop/vllm.err >&2 || true
    tail -n 50 logs_random_seq_3hop/vllm.out >&2 || true
    exit 1
  fi
  if curl -sf "$OPENAI_API_BASE/models" >/dev/null 2>&1; then
    echo "[ok] vLLM API is up" >&2; break
  fi
  tries=$((tries+1))
  if [[ "$tries" -ge "$VLLM_WAIT_TRIES" ]]; then
    echo "[error] vLLM not ready in time. Recent logs:" >&2
    tail -n 100 logs_random_seq_3hop/vllm.err >&2 || true
    tail -n 50 logs_random_seq_3hop/vllm.out >&2 || true
    exit 1
  fi
  sleep 2
done

# -------- helper: compute d1,d2,d3 from linear budget (3-hop) --------
calc_d1_d2_d3() {
  local L="$1" K="$2" D1_MAX="$3" D2_MAX="$4" D3_MAX="$5"
  python - "$L" "$K" "$D1_MAX" "$D2_MAX" "$D3_MAX" <<'PY'
import math, sys
L      = int(sys.argv[1])
K      = int(sys.argv[2])
D1_MAX = int(sys.argv[3])
D2_MAX = int(sys.argv[4])
D3_MAX = int(sys.argv[5])
W = max(1, K*L)
# For 3-hop: cube root distribution
d_base = max(1, int(W ** (1.0/3.0)))
d1 = min(d_base, D1_MAX)
d2 = min(d_base, D2_MAX)
d3 = min(d_base, D3_MAX)
print(d1)   # line 1
print(d2)   # line 2
print(d3)   # line 3
print(W)    # line 4
PY
}

# ----------------- Run batch -----------------
echo "[run] 3-hop Linear-budget sweep with K=${K}; fixed TOPK=${TOPK_FIXED}; limits: 1-15 plus 16 20 25 30 50 100" >&2

for L in "${ALL_LIMITS[@]}"; do
  # derive d1,d2,d3 from linear budget
  readarray -t _nums < <(calc_d1_d2_d3 "$L" "$K" "$D1_MAX" "$D2_MAX" "$D3_MAX")
  D1="${_nums[0]}"
  D2="${_nums[1]}"
  D3="${_nums[2]}"
  W="${_nums[3]}"

  run_prefix="[run]"; ok_prefix="[ok]"; metrics_prefix="[metrics]"; error_prefix="[error]"
  if (( L > 15 )); then
    run_prefix="[run][high]"; ok_prefix="[ok][high]"
    metrics_prefix="[metrics][high]"; error_prefix="[error][high]"
  fi

  OUTPUT_PATH="${BASE_OUTPUT_DIR}/answers_random_seq_3hop_limit${L}.jsonl"
  echo "${run_prefix} limit=${L}, d1=${D1}, d2=${D2}, d3=${D3}, W~${W} -> ${OUTPUT_PATH}" >&2

  python examples/retrieve/batch_answers_from_dataset_random_seq_3hop.py \
    --input "$DATASET_PATH" \
    --output "$OUTPUT_PATH" \
    --aggregate min \
    --normalize none \
    --d1 "${D1}" \
    --d2 "${D2}" \
    --d3 "${D3}" \
    --limit "$L" \
    --topk "$TOPK_FIXED" \
    --max-rows "${BATCH_MAX_ROWS:-0}" \
    --uri "$NEO4J_URI" --user "$NEO4J_USER" --password "$NEO4J_PASSWORD" --database "$NEO4J_DATABASE" \
    --provider vllm --model "$MODEL" --base-url "$OPENAI_API_BASE" --api-key "$OPENAI_API_KEY" \
    || { echo "${error_prefix} Batch failed for limit=$L" >&2; exit 1; }

  echo "${ok_prefix} Answers written to $OUTPUT_PATH" >&2
  echo "[sample] (limit=$L) Last 3:" >&2
  tail -n 3 "$OUTPUT_PATH" || true

  METRICS_PATH="${OUTPUT_PATH}.metrics.json"
  if [[ -f "$METRICS_PATH" ]]; then
    echo "${metrics_prefix} (limit=$L) Summary:" >&2
    jq '{rows_processed, elapsed_seconds, throughput_rows_per_sec} // .' "$METRICS_PATH" 2>/dev/null || cat "$METRICS_PATH" || true
  fi
done

echo "[done] 3-hop Linear-budget sweep complete" >&2
