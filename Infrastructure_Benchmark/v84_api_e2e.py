from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], pct: float):
    if not values:
        return None

    ordered = sorted(values)
    rank = max(1, int(round((pct / 100.0) * len(ordered))))
    rank = min(rank, len(ordered))
    return round(ordered[rank - 1], 4)


async def send_one(
    client: httpx.AsyncClient,
    base_url: str,
    file_path: Path,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        start = time.perf_counter()

        try:
            content_type = (
                mimetypes.guess_type(file_path.name)[0]
                or "application/octet-stream"
            )

            with file_path.open("rb") as f:
                files = {
                    "file": (
                        file_path.name,
                        f,
                        content_type,
                    )
                }

                response = await client.post(
                    base_url.rstrip("/") + "/predict",
                    files=files,
                )

            latency = time.perf_counter() - start

            result = {
                "file": file_path.name,
                "status_code": response.status_code,
                "latency_s": round(latency, 4),
                "ok": response.status_code == 200,
            }

            if response.status_code == 200:
                try:
                    payload = response.json()
                    result["predicted_competencies"] = payload.get(
                        "proposed_competencies", []
                    )
                    result["difficulty_level"] = payload.get(
                        "difficulty_level"
                    )
                    result["response_size_bytes"] = len(response.content)
                except Exception as exc:
                    result["ok"] = False
                    result["error"] = f"JSON parse error: {exc}"
            else:
                result["error"] = response.text[:500]

            return result

        except Exception as exc:
            latency = time.perf_counter() - start
            return {
                "file": file_path.name,
                "status_code": None,
                "latency_s": round(latency, 4),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def run_level(
    base_url: str,
    files: list[Path],
    concurrency: int,
    timeout: float,
):
    semaphore = asyncio.Semaphore(concurrency)

    limits = httpx.Limits(
        max_connections=concurrency + 4,
        max_keepalive_connections=concurrency + 4,
    )

    timeout_config = httpx.Timeout(
        timeout,
        connect=15.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout_config,
        limits=limits,
    ) as client:

        start = time.perf_counter()

        results = await asyncio.gather(
            *[
                send_one(
                    client,
                    base_url,
                    file_path,
                    semaphore,
                )
                for file_path in files
            ]
        )

        wall_time = time.perf_counter() - start

    successful = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    latencies = [
        r["latency_s"]
        for r in successful
    ]

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "success": len(successful),
        "errors": len(failed),
        "error_rate": round(
            len(failed) / len(results),
            4,
        ) if results else 0,
        "wall_time_s": round(wall_time, 4),
        "throughput_req_s": round(
            len(results) / wall_time,
            4,
        ) if wall_time > 0 else 0,
        "latency_avg_s": round(
            statistics.mean(latencies),
            4,
        ) if latencies else None,
        "latency_p50_s": percentile(
            latencies,
            50,
        ),
        "latency_p95_s": percentile(
            latencies,
            95,
        ),
        "latency_p99_s": percentile(
            latencies,
            99,
        ),
        "latency_min_s": round(
            min(latencies),
            4,
        ) if latencies else None,
        "latency_max_s": round(
            max(latencies),
            4,
        ) if latencies else None,
        "results": results,
    }


def print_summary(report):
    print()
    print(
        f"{'CONC':>5} "
        f"{'REQ':>5} "
        f"{'OK':>5} "
        f"{'ERR':>5} "
        f"{'ERR%':>7} "
        f"{'AVG(s)':>9} "
        f"{'P50(s)':>9} "
        f"{'P95(s)':>9} "
        f"{'P99(s)':>9} "
        f"{'THR(req/s)':>12}"
    )

    print("-" * 100)

    for level in report["levels"]:
        def fmt(value):
            return "n/a" if value is None else f"{value:.3f}"

        print(
            f"{level['concurrency']:>5} "
            f"{level['requests']:>5} "
            f"{level['success']:>5} "
            f"{level['errors']:>5} "
            f"{level['error_rate'] * 100:>6.1f}% "
            f"{fmt(level['latency_avg_s']):>9} "
            f"{fmt(level['latency_p50_s']):>9} "
            f"{fmt(level['latency_p95_s']):>9} "
            f"{fmt(level['latency_p99_s']):>9} "
            f"{level['throughput_req_s']:>12.4f}"
        )


async def main_async(args):
    evaluation_dir = Path(args.data_dir)

    all_files = sorted(
        [
            p
            for p in evaluation_dir.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in {
                ".pptx",
                ".ipynb",
                ".xlsx",
                ".md",
            }
            and p.name != "README_STUDENTS.md"
        ]
    )

    if len(all_files) != 12:
        print(
            f"WARNING: expected 12 evaluation files, "
            f"found {len(all_files)}"
        )

    print("Evaluation files:")
    for p in all_files:
        print(f"  - {p.name}")

    report = {
        "timestamp": int(time.time()),
        "base_url": args.base_url,
        "endpoint": "/predict",
        "data_dir": str(evaluation_dir),
        "file_count": len(all_files),
        "concurrency_levels": args.concurrency,
        "timeout_s": args.timeout,
        "levels": [],
    }

    for concurrency in args.concurrency:
        print()
        print(
            f"===== CONCURRENCY {concurrency} ====="
        )

        level = await run_level(
            base_url=args.base_url,
            files=all_files,
            concurrency=concurrency,
            timeout=args.timeout,
        )

        report["levels"].append(level)

        print(
            f"completed: "
            f"{level['success']}/{level['requests']} "
            f"successful | "
            f"wall={level['wall_time_s']}s | "
            f"throughput={level['throughput_req_s']} req/s"
        )

    print_summary(report)

    output = Path(args.out)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved report: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark for deployed V8.4 API"
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:18000",
    )

    parser.add_argument(
        "--data-dir",
        default="Model/evaluation_data",
    )

    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="comma-separated concurrency levels",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--out",
        default="Infrastructure_Benchmark/results/v84_api_e2e.json",
    )

    args = parser.parse_args()

    args.concurrency = [
        int(x.strip())
        for x in args.concurrency.split(",")
        if x.strip()
    ]

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
