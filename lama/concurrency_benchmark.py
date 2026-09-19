#!/usr/bin/env python3
"""
Infrastructure benchmark for the Content Tagging & Competency Mapping model.

Measures exactly what the capstone's infrastructure section asks for:
  - TTFT (time to first token, via streaming)
  - Total latency per request
  - Tokens/sec
  - Throughput (requests/sec)
  - Behavior at concurrency = 1, 2, 5, 10
  - p95 latency, error rate
  - Cost per request / per course / estimated monthly cost
  - A snapshot of GPU utilization + VRAM (if run on the GPU host itself)

Works against any OpenAI-compatible endpoint (vLLM's default server API),
so it runs unchanged whether you're hitting a local vLLM instance on the
AIDC GPUs or a hosted API.

USAGE
-----
    python3 concurrency_benchmark.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --payloads benchmark_payloads.json \
        --gpu-hourly-cost 1.80 \
        --courses-per-month 200

Requires: pip install httpx --break-system-packages
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    raise SystemExit("Run: pip install httpx --break-system-packages")


CONCURRENCY_LEVELS = [1, 2, 5, 10]
REQUESTS_PER_LEVEL = 20  # cycles through the payload set to get stable stats


def load_payloads(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def gpu_snapshot():
    """Best-effort nvidia-smi snapshot. Returns None if not run on a GPU host."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        ).decode().strip()
        util, used, total = [x.strip() for x in out.split(",")]
        return {"gpu_util_pct": float(util), "vram_used_mb": float(used), "vram_total_mb": float(total)}
    except Exception:
        return None


async def send_one(client: httpx.AsyncClient, base_url: str, model: str, prompt: str):
    """
    Sends one streaming chat completion request.
    Returns dict with ttft_s, total_latency_s, output_tokens, error (or None).
    """
    t0 = time.perf_counter()
    ttft = None
    output_chars = 0
    usage = None
    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 400,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    output_chars += len(delta.get("content") or "")
                if chunk.get("usage"):
                    usage = chunk["usage"]
        total = time.perf_counter() - t0
        return {
            "ttft_s": ttft if ttft is not None else total,
            "total_latency_s": total,
            "output_tokens": (usage or {}).get("completion_tokens") or round(output_chars / 4),
            "input_tokens": (usage or {}).get("prompt_tokens"),
            "error": None,
        }
    except Exception as e:
        return {"ttft_s": None, "total_latency_s": time.perf_counter() - t0,
                "output_tokens": 0, "input_tokens": None, "error": str(e)}


async def run_concurrency_level(base_url, model, payloads, concurrency, n_requests):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def bound_call(client, prompt):
        async with sem:
            return await send_one(client, base_url, model, prompt)

    prompts = [payloads[i % len(payloads)]["prompt"] for i in range(n_requests)]

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [bound_call(client, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t0

    return results, wall_time


def summarize(results, wall_time):
    ok = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    latencies = [r["total_latency_s"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    out_tokens = sum(r["output_tokens"] for r in ok)

    def p95(vals):
        if not vals:
            return None
        s = sorted(vals)
        idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
        return s[idx]

    return {
        "n": len(results),
        "errors": len(errors),
        "error_rate_pct": round(100 * len(errors) / len(results), 1) if results else 0,
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else None,
        "p95_latency_s": round(p95(latencies), 3) if latencies else None,
        "avg_ttft_s": round(statistics.mean(ttfts), 3) if ttfts else None,
        "throughput_req_s": round(len(ok) / wall_time, 3) if wall_time else None,
        "tokens_per_sec": round(out_tokens / wall_time, 1) if wall_time else None,
        "wall_time_s": round(wall_time, 2),
    }


async def main_async(args):
    payloads = load_payloads(args.payloads)
    print(f"Loaded {len(payloads)} real content payloads "
          f"(avg ~{round(sum(p['approx_input_tokens'] for p in payloads)/len(payloads))} input tokens each)\n")

    gpu_before = gpu_snapshot()
    if gpu_before:
        print(f"GPU before run: {gpu_before['gpu_util_pct']}% util, "
              f"{gpu_before['vram_used_mb']:.0f}/{gpu_before['vram_total_mb']:.0f} MB VRAM\n")
    else:
        print("(nvidia-smi not available here -- run this script on the GPU host itself "
              "to capture GPU utilization / VRAM.)\n")

    table_rows = []
    for c in CONCURRENCY_LEVELS:
        print(f"--- Concurrency = {c} ---")
        results, wall_time = await run_concurrency_level(args.base_url, args.model, payloads, c, REQUESTS_PER_LEVEL)
        summary = summarize(results, wall_time)
        summary["concurrency"] = c
        gpu_after = gpu_snapshot()
        summary["gpu_util_pct"] = gpu_after["gpu_util_pct"] if gpu_after else None
        summary["vram_used_mb"] = gpu_after["vram_used_mb"] if gpu_after else None
        table_rows.append(summary)
        print(json.dumps(summary, indent=2))
        if summary["errors"]:
            for r in results:
                if r["error"]:
                    print("  error sample:", r["error"][:200])
        print()

    # ---- Part 21 table ----
    print("=" * 70)
    print("PART 21 -- Concurrency table")
    print(f"{'Concurrency':<12}{'Avg latency':<14}{'p95 latency':<14}{'Throughput':<14}{'Errors':<8}")
    for r in table_rows:
        print(f"{r['concurrency']:<12}{str(r['avg_latency_s'])+'s':<14}{str(r['p95_latency_s'])+'s':<14}"
              f"{str(r['throughput_req_s'])+'/s':<14}{r['error_rate_pct']}%")

    # ---- Part 22 cost calc ----
    print("\n" + "=" * 70)
    print("PART 22 -- Cost per course (self-hosted GPU)")
    best = table_rows[0]  # single-request latency is the fair per-course cost basis
    if best["avg_latency_s"]:
        courses_per_hour = 3600 / best["avg_latency_s"]
        gpu_cost_per_course = args.gpu_hourly_cost / courses_per_hour
        monthly_cost = gpu_cost_per_course * args.courses_per_month
        print(f"GPU hourly cost:            ${args.gpu_hourly_cost:.2f}/hr")
        print(f"Courses/hour @ concurrency 1: {courses_per_hour:.1f}")
        print(f"Cost per course:            ${gpu_cost_per_course:.5f}")
        print(f"Estimated monthly cost ({args.courses_per_month} courses/mo): ${monthly_cost:.2f}")

    # ---- write CSV ----
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("concurrency,avg_latency_s,p95_latency_s,avg_ttft_s,throughput_req_s,tokens_per_sec,error_rate_pct,gpu_util_pct,vram_used_mb\n")
        for r in table_rows:
            f.write(f"{r['concurrency']},{r['avg_latency_s']},{r['p95_latency_s']},{r['avg_ttft_s']},"
                    f"{r['throughput_req_s']},{r['tokens_per_sec']},{r['error_rate_pct']},"
                    f"{r['gpu_util_pct']},{r['vram_used_mb']}\n")
    print(f"\nSaved results table to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible base URL (vLLM default)")
    ap.add_argument("--model", required=True, help="Model name as registered with the server")
    ap.add_argument("--payloads", default="benchmark_payloads.json", help="Path to benchmark_payloads.json")
    ap.add_argument("--gpu-hourly-cost", type=float, default=1.80, help="$/hr for the GPU used (cloud-equivalent rate)")
    ap.add_argument("--courses-per-month", type=int, required=True,
                     help="Expected monthly course volume -- ask the use-case owner, do not invent this")
    ap.add_argument("--output", default="concurrency_results.csv")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
