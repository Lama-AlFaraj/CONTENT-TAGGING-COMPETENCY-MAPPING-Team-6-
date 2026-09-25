#!/bin/bash

set -e

PROM="http://localhost:9091/api/v1/query"
OUT="Infrastructure_Benchmark/results/llm_performance_benchmark.csv"

query() {
    curl -sG "$PROM" \
        --data-urlencode "query=$1" |
        python3 -c '
import sys,json
d=json.load(sys.stdin)
try:
    print(d["data"]["result"][0]["value"][1])
except:
    print("0")
'
}

MODEL='Qwen/Qwen2.5-7B-Instruct-AWQ'

echo "Taking pre-benchmark metric snapshot..."

GEN_BEFORE=$(query 'vllm:generation_tokens_total{model_name="'$MODEL'"}')
REQ_BEFORE=$(query 'vllm:request_generation_tokens_count{model_name="'$MODEL'"}')
TTFT_SUM_BEFORE=$(query 'vllm:time_to_first_token_seconds_sum{model_name="'$MODEL'"}')
TTFT_COUNT_BEFORE=$(query 'vllm:time_to_first_token_seconds_count{model_name="'$MODEL'"}')
ITL_SUM_BEFORE=$(query 'vllm:inter_token_latency_seconds_sum{model_name="'$MODEL'"}')
ITL_COUNT_BEFORE=$(query 'vllm:inter_token_latency_seconds_count{model_name="'$MODEL'"}')

START=$(date +%s.%N)

echo ""
echo "=========================================="
echo "RUNNING V8.4 C=4 BENCHMARK"
echo "=========================================="
echo ""

python3 Infrastructure_Benchmark/v84_api_e2e.py \
    --base-url http://localhost:8002 \
    --concurrency 4 \
    --out Infrastructure_Benchmark/results/v84_llm_metrics_c4.json

END=$(date +%s.%N)

echo ""
echo "Taking post-benchmark metric snapshot..."

GEN_AFTER=$(query 'vllm:generation_tokens_total{model_name="'$MODEL'"}')
REQ_AFTER=$(query 'vllm:request_generation_tokens_count{model_name="'$MODEL'"}')
TTFT_SUM_AFTER=$(query 'vllm:time_to_first_token_seconds_sum{model_name="'$MODEL'"}')
TTFT_COUNT_AFTER=$(query 'vllm:time_to_first_token_seconds_count{model_name="'$MODEL'"}')
ITL_SUM_AFTER=$(query 'vllm:inter_token_latency_seconds_sum{model_name="'$MODEL'"}')
ITL_COUNT_AFTER=$(query 'vllm:inter_token_latency_seconds_count{model_name="'$MODEL'"}')

python3 - "$START" "$END" \
    "$GEN_BEFORE" "$GEN_AFTER" \
    "$REQ_BEFORE" "$REQ_AFTER" \
    "$TTFT_SUM_BEFORE" "$TTFT_SUM_AFTER" \
    "$TTFT_COUNT_BEFORE" "$TTFT_COUNT_AFTER" \
    "$ITL_SUM_BEFORE" "$ITL_SUM_AFTER" \
    "$ITL_COUNT_BEFORE" "$ITL_COUNT_AFTER" \
    "$OUT" <<'PY'
import sys
import csv

(
    start, end,
    gen_b, gen_a,
    req_b, req_a,
    ttft_sum_b, ttft_sum_a,
    ttft_count_b, ttft_count_a,
    itl_sum_b, itl_sum_a,
    itl_count_b, itl_count_a,
    out
) = sys.argv[1:]

start = float(start)
end = float(end)
duration = end - start

gen_delta = float(gen_a) - float(gen_b)
req_delta = float(req_a) - float(req_b)

ttft_sum_delta = float(ttft_sum_a) - float(ttft_sum_b)
ttft_count_delta = float(ttft_count_a) - float(ttft_count_b)

itl_sum_delta = float(itl_sum_a) - float(itl_sum_b)
itl_count_delta = float(itl_count_a) - float(itl_count_b)

avg_ttft = (
    ttft_sum_delta / ttft_count_delta
    if ttft_count_delta > 0 else 0
)

avg_itl = (
    itl_sum_delta / itl_count_delta
    if itl_count_delta > 0 else 0
)

tokens_per_sec = (
    gen_delta / duration
    if duration > 0 else 0
)

tokens_per_request = (
    gen_delta / req_delta
    if req_delta > 0 else 0
)

rows = [
    ["benchmark_duration_s", round(duration, 4)],
    ["generation_tokens", round(gen_delta, 2)],
    ["generation_requests", round(req_delta, 2)],
    ["tokens_per_second", round(tokens_per_sec, 4)],
    ["generation_tokens_per_request", round(tokens_per_request, 4)],
    ["avg_ttft_s", round(avg_ttft, 6)],
    ["avg_ttft_ms", round(avg_ttft * 1000, 3)],
    ["avg_inter_token_latency_s", round(avg_itl, 6)],
    ["avg_inter_token_latency_ms", round(avg_itl * 1000, 3)],
]

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows(rows)

print()
print("==========================================")
print("LLM BENCHMARK RESULTS")
print("==========================================")
for k, v in rows:
    print(f"{k}: {v}")

print()
print(f"Saved: {out}")
PY
