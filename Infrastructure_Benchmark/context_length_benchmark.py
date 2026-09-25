"""
Context-length benchmark for the BeamData V8.4 /predict API.

What this measures, that the existing benchmarks (bench.py, v84_api_e2e.py)
do not:
  - Latency and error rate as a function of INPUT LENGTH specifically
    (short / medium / long buckets), not just as a function of concurrency.
  - Whether the deployed --max-model-len is actually sufficient for the
    longest real files, and how latency scales with prompt size.

It reuses the same 12-file evaluation set already in the repo
(Model/evaluation_data/) so the results are directly comparable with the
existing concurrency benchmark and the model-quality evaluation.

Usage
-----
    python context_length_benchmark.py \
        --base-url http://127.0.0.1:8002 \
        --data-dir Model/evaluation_data \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --out context_length_benchmark_results.json

Requires: httpx, transformers (for the tokenizer used only to bucket files
by real token count -- pip install "transformers" if not already present;
falls back to a whitespace-word-count heuristic if unavailable).

This script only sends requests -- it does not launch vLLM or the API.
Run it against a live BeamData API instance (local port-forward or
in-cluster), the same way v84_api_e2e.py is run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import statistics
import time
from pathlib import Path

import httpx

try:
    from transformers import AutoTokenizer
    _HAS_TOKENIZER = True
except ImportError:
    _HAS_TOKENIZER = False


def count_tokens(text_or_bytes, tokenizer) -> int:
    if isinstance(text_or_bytes, int):
        # Raw byte count from a binary file (.pptx/.xlsx) -- rough proxy:
        # extracted text from these formats is typically a small fraction
        # of the raw file size (XML markup, embedded media, styles, etc.
        # inflate the byte size well beyond the actual text). ~1 token per
        # 8 raw bytes is a conservative, crash-proof estimate.
        return max(1, text_or_bytes // 8)
    text = text_or_bytes
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    # Fallback heuristic: ~0.75 tokens per whitespace word for English/code mixes.
    return int(len(text.split()) / 0.75)


def bucket_for(token_count: int) -> str:
    if token_count < 2000:
        return "short (<2k tokens)"
    if token_count < 6000:
        return "medium (2k-6k tokens)"
    if token_count < 12000:
        return "long (6k-12k tokens)"
    return "very_long (>=12k tokens)"


def percentile(values: list[float], pct: float):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(round((pct / 100.0) * len(ordered))))
    rank = min(rank, len(ordered))
    return round(ordered[rank - 1], 4)


async def send_one(client: httpx.AsyncClient, base_url: str, file_path: Path):
    start = time.perf_counter()
    try:
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, content_type)}
            response = await client.post(base_url.rstrip("/") + "/predict", files=files)
        latency = time.perf_counter() - start
        ok = response.status_code == 200
        result = {
            "file": file_path.name,
            "status_code": response.status_code,
            "latency_s": round(latency, 4),
            "ok": ok,
        }
        if not ok:
            result["error"] = response.text[:500]
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "file": file_path.name,
            "status_code": None,
            "latency_s": round(time.perf_counter() - start, 4),
            "ok": False,
            "error": str(exc),
        }


_MAX_CHARS_FOR_TOKENIZER = 200_000  # guard against pathological regex backtracking


def _pptx_text(path: Path) -> str:
    """Same extraction as Model/v8/api.py -> extract_pptx (text only, no media)."""
    from pptx import Presentation
    parts = []
    for i, slide in enumerate(Presentation(str(path)).slides, start=1):
        texts = [sh.text.strip() for sh in slide.shapes
                 if hasattr(sh, "text") and sh.text.strip()]
        if texts:
            parts.append(f"[SLIDE {i}]\n" + "\n".join(texts))
    return "\n\n".join(parts)


def _xlsx_text(path: Path) -> str:
    """Same extraction as Model/v8/api.py -> extract_xlsx."""
    from openpyxl import load_workbook
    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        rows = [" | ".join(str(v) for v in r if v is not None)
                for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if r]
        if rows:
            parts.append(f"[SHEET: {ws.title}]\n" + "\n".join(rows))
    return "\n\n".join(parts)



def extract_text_for_bucketing(path: Path):
    """Best-effort raw text pull just for token counting (not the real
    extraction logic used by the API -- that happens server-side).

    Returns either a string (to be tokenized normally) or an int
    (a byte count, when the file type isn't plain text -- this is
    converted to an estimated token count directly, WITHOUT ever
    constructing a giant filler string and feeding it to the
    tokenizer, which can hang/crash the tokenizer's regex engine on
    large inputs).
    """
    suffix = path.suffix.lower()
    try:
        if suffix in {".md", ".txt", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")[:_MAX_CHARS_FOR_TOKENIZER]
        if suffix == ".ipynb":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            text = "\n".join(
                "".join(cell.get("source", []))
                for cell in data.get("cells", [])
            )
            return text[:_MAX_CHARS_FOR_TOKENIZER]
        if suffix == ".pptx":
            return _pptx_text(path)[:_MAX_CHARS_FOR_TOKENIZER]
        if suffix == ".xlsx":
            return _xlsx_text(path)[:_MAX_CHARS_FOR_TOKENIZER]
        # Unknown type: byte-size proxy (see count_tokens's int branch).
        return path.stat().st_size
    except Exception:  # noqa: BLE001
        return path.stat().st_size


async def main_async(args):
    tokenizer = None
    if _HAS_TOKENIZER:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not load tokenizer {args.tokenizer}: {exc}. "
                  f"Falling back to a word-count heuristic.")

    data_dir = Path(args.data_dir)
    exclude = set(args.exclude)
    files = sorted(p for p in data_dir.iterdir()
                   if p.is_file() and p.name not in exclude)
    if not files:
        raise SystemExit(f"No files found under {data_dir}")

    file_info = []
    for f in files:
        text = extract_text_for_bucketing(f)
        tokens = count_tokens(text, tokenizer)
        file_info.append({"file": f, "tokens": tokens, "bucket": bucket_for(tokens)})

    print("File -> estimated input tokens -> bucket")
    for fi in file_info:
        print(f"  {fi['file'].name:55s} ~{fi['tokens']:6d} tok  [{fi['bucket']}]")

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        raw_results = []
        for fi in file_info:
            for rep in range(1, args.repeat + 1):
                r = await send_one(client, args.base_url, fi["file"])
                r["estimated_tokens"] = fi["tokens"]
                r["bucket"] = fi["bucket"]
                r["repeat"] = rep
                raw_results.append(r)
                print(f"    -> {r['file']} (run {rep}/{args.repeat}): "
                      f"{'OK' if r['ok'] else 'FAIL'} in {r['latency_s']}s")

    buckets: dict[str, list[dict]] = {}
    for r in raw_results:
        buckets.setdefault(r["bucket"], []).append(r)

    summary = {}
    for bucket, results in buckets.items():
        latencies = [r["latency_s"] for r in results if r["ok"]]
        errors = sum(1 for r in results if not r["ok"])
        summary[bucket] = {
            "n_files": len({r["file"] for r in results}),
            "n_requests": len(results),
            "errors": errors,
            "error_rate": round(errors / len(results), 4),
            "avg_latency_s": round(statistics.mean(latencies), 4) if latencies else None,
            "p50_latency_s": percentile(latencies, 50),
            "p95_latency_s": percentile(latencies, 95),
            "min_tokens": min(r["estimated_tokens"] for r in results),
            "max_tokens": max(r["estimated_tokens"] for r in results),
        }

    output = {
        "timestamp": int(time.time()),
        "base_url": args.base_url,
        "data_dir": str(data_dir),
        "tokenizer_used": args.tokenizer if tokenizer is not None else "heuristic (word-count based)",
        "per_file": raw_results,
        "by_bucket": summary,
    }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print("\nSummary by input-length bucket:")
    for bucket, s in summary.items():
        print(f"  {bucket}: n={s['n_files']}, errors={s['errors']}, "
              f"reqs={s['n_requests']}, avg={s['avg_latency_s']}s, p95={s['p95_latency_s']}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8002")
    parser.add_argument("--data-dir", default="Model/evaluation_data")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default="context_length_benchmark_results.json")
    parser.add_argument("--repeat", type=int, default=1,
                        help="requests per file (use 3 for stabler p50/p95; "
                             "note repeats hit the prefix cache if it is ON)")
    parser.add_argument("--exclude", nargs="*",
                        default=["README_STUDENTS.md", "model_output_template.csv"],
                        help="files in --data-dir that are not part of the 12-file eval set")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
