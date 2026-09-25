import csv
from pathlib import Path

base = Path("Infrastructure_Benchmark/results")

api_file = base / "api_performance_benchmark.csv"
llm_file = base / "llm_performance_benchmark.csv"
gpu_file = base / "gpu_monitoring_results.csv"
out_file = base / "final_benchmark_report.csv"

rows = []

# API benchmark results
with open(api_file) as f:
    api = list(csv.DictReader(f))

for r in api:
    rows.append({
        "category": "API",
        "metric": f"Concurrency {r['concurrency']} - Avg Latency",
        "value": r["avg_latency_s"],
        "unit": "seconds"
    })
    rows.append({
        "category": "API",
        "metric": f"Concurrency {r['concurrency']} - P50 Latency",
        "value": r["p50_s"],
        "unit": "seconds"
    })
    rows.append({
        "category": "API",
        "metric": f"Concurrency {r['concurrency']} - P95 Latency",
        "value": r["p95_s"],
        "unit": "seconds"
    })
    rows.append({
        "category": "API",
        "metric": f"Concurrency {r['concurrency']} - P99 Latency",
        "value": r["p99_s"],
        "unit": "seconds"
    })
    rows.append({
        "category": "API",
        "metric": f"Concurrency {r['concurrency']} - Throughput",
        "value": r["throughput_req_s"],
        "unit": "requests/sec"
    })
    rows.append({
        "category": "API",
        "metric": f"Concurrency {r['concurrency']} - Error Rate",
        "value": r["error_rate"],
        "unit": "ratio"
    })

# LLM benchmark
with open(llm_file) as f:
    llm = {r["metric"]: r["value"] for r in csv.DictReader(f)}

llm_metrics = [
    ("benchmark_duration_s", "Benchmark Duration", "seconds"),
    ("generation_tokens", "Generated Tokens", "tokens"),
    ("tokens_per_second", "Generation Throughput", "tokens/sec"),
    ("generation_tokens_per_request", "Generated Tokens per Request", "tokens/request"),
    ("avg_ttft_ms", "Average TTFT", "ms"),
    ("avg_inter_token_latency_ms", "Average Inter-token Latency", "ms"),
]

for key, name, unit in llm_metrics:
    rows.append({
        "category": "LLM",
        "metric": name,
        "value": llm[key],
        "unit": unit
    })

# GPU monitoring
with open(gpu_file) as f:
    for r in csv.DictReader(f):
        rows.append({
            "category": "GPU",
            "metric": r["metric"],
            "value": r["value"],
            "unit": r["unit"]
        })

# System metadata
system_rows = [
    ("Model", "Qwen/Qwen2.5-7B-Instruct-AWQ", ""),
    ("Serving Engine", "vLLM", ""),
    ("API", "FastAPI", ""),
    ("GPU", "NVIDIA RTX A6000", ""),
    ("Deployment", "Kubernetes", ""),
]

for metric, value, unit in system_rows:
    rows.append({
        "category": "System",
        "metric": metric,
        "value": value,
        "unit": unit
    })

with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["category", "metric", "value", "unit"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Created: {out_file}")
print(f"Rows: {len(rows)}")
