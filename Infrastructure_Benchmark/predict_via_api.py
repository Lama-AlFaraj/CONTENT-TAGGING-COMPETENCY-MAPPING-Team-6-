"""
Send the evaluation files to any BeamData-style /predict endpoint and save the
answers in the same CSV format as Model/v8/results/v8_predictions.csv, so the
result can be scored with evaluate_v8.py (PRED_PATH=... python3 evaluate_v8.py).

Works for the AWQ API, the FP16 comparison API, or any other backend that
exposes POST /predict.

    python3 predict_via_api.py --base-url http://127.0.0.1:8012 \
        --data-dir Model/evaluation_data --out Model/v8/results/fp16_predictions.csv
"""
import argparse, csv, json, mimetypes, time
from pathlib import Path
import httpx

COLUMNS = ["file_name", "predicted_tags", "proposed_competencies", "difficulty_level",
           "confidence", "notes", "chunk_count", "retrieval_candidates", "latency_s", "status"]
SUPPORTED = {".pptx", ".ipynb", ".xlsx", ".md", ".txt", ".csv"}
NOT_EVAL = {"README_STUDENTS.md", "model_output_template.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data-dir", default="Model/evaluation_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.data_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED and p.name not in NOT_EVAL)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(timeout=a.timeout) as client:
        for f in files:
            t0 = time.perf_counter()
            row = {c: "" for c in COLUMNS}
            row["file_name"] = f.name
            try:
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                with f.open("rb") as fh:
                    r = client.post(a.base_url.rstrip("/") + "/predict", files={"file": (f.name, fh, ctype)})
                if r.status_code == 200:
                    d = r.json()
                    row.update(
                        predicted_tags=json.dumps(d.get("predicted_tags", []), ensure_ascii=False),
                        proposed_competencies=json.dumps(d.get("proposed_competencies", []), ensure_ascii=False),
                        difficulty_level=d.get("difficulty_level", ""),
                        confidence=d.get("confidence", ""),
                        notes=d.get("notes", ""),
                        chunk_count=d.get("chunk_count", ""),
                        retrieval_candidates=json.dumps(d.get("retrieval_candidates", []), ensure_ascii=False),
                        status="success")
                else:
                    row["status"] = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"error: {exc}"
            row["latency_s"] = round(time.perf_counter() - t0, 3)
            rows.append(row)
            print(f"{f.name:55s} {row['status'][:40]:40s} {row['latency_s']}s")

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader(); w.writerows(rows)
    ok = sum(r["status"] == "success" for r in rows)
    lat = [r["latency_s"] for r in rows if r["status"] == "success"]
    print(f"\nWrote {a.out}: {ok}/{len(rows)} succeeded"
          + (f", avg latency {sum(lat)/len(lat):.2f}s" if lat else ""))


if __name__ == "__main__":
    main()
