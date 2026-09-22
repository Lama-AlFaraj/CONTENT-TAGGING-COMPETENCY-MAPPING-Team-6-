import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OPENAI_FILE = ROOT / "benchmark" / "results" / "openai_evaluation.json"
V84_FILE = ROOT / "benchmark" / "results" / "v84_api_e2e_c1_clean.json"

RESULTS_DIR = ROOT / "benchmark" / "results"
JSON_OUT = RESULTS_DIR / "openai_vs_v84_comparison.json"
MD_OUT = ROOT / "benchmark" / "OPENAI_VS_V84_COMPARISON.md"


def normalize(items):
    return {str(x).strip().lower() for x in (items or [])}


openai_data = json.loads(OPENAI_FILE.read_text())
v84_data = json.loads(V84_FILE.read_text())

openai_results = {
    r["file_name"]: r
    for r in openai_data["results"]
}

v84_level = v84_data["levels"][0]
v84_results = {
    r["file"]: r
    for r in v84_level["results"]
}

files = sorted(set(openai_results) & set(v84_results))

comparisons = []

for filename in files:
    o = openai_results[filename]
    v = v84_results[filename]

    o_comp = o.get("proposed_competencies", [])
    v_comp = v.get("predicted_competencies", [])

    o_set = normalize(o_comp)
    v_set = normalize(v_comp)

    intersection = o_set & v_set
    union = o_set | v_set

    if union:
        jaccard = len(intersection) / len(union)
    else:
        jaccard = 1.0

    exact_competency_match = o_set == v_set
    difficulty_match = (
        o.get("difficulty_level") == v.get("difficulty_level")
    )

    comparisons.append({
        "file": filename,
        "openai_competencies": o_comp,
        "v84_competencies": v_comp,
        "shared_competencies": sorted(intersection),
        "openai_only": sorted(o_set - v_set),
        "v84_only": sorted(v_set - o_set),
        "competency_exact_match": exact_competency_match,
        "competency_jaccard": round(jaccard, 4),
        "openai_difficulty": o.get("difficulty_level"),
        "v84_difficulty": v.get("difficulty_level"),
        "difficulty_match": difficulty_match,
        "openai_confidence": o.get("confidence"),
        "openai_latency_s": o.get("latency_s"),
        "v84_latency_s": v.get("latency_s"),
        "latency_difference_s": round(
            o.get("latency_s", 0) - v.get("latency_s", 0), 4
        ),
    })


n = len(comparisons)

exact_matches = sum(
    x["competency_exact_match"] for x in comparisons
)

difficulty_matches = sum(
    x["difficulty_match"] for x in comparisons
)

avg_jaccard = (
    sum(x["competency_jaccard"] for x in comparisons) / n
    if n else 0
)

avg_openai_latency = (
    sum(x["openai_latency_s"] for x in comparisons) / n
    if n else 0
)

avg_v84_latency = (
    sum(x["v84_latency_s"] for x in comparisons) / n
    if n else 0
)

json_output = {
    "comparison_type": "model_to_model_agreement",
    "note": (
        "This comparison measures agreement and differences between "
        "OpenAI and V8.4. It is not an accuracy benchmark because "
        "no gold labels were provided."
    ),
    "files_compared": n,
    "competency_exact_match_count": exact_matches,
    "competency_exact_match_rate": round(exact_matches / n, 4) if n else 0,
    "difficulty_match_count": difficulty_matches,
    "difficulty_match_rate": round(difficulty_matches / n, 4) if n else 0,
    "average_competency_jaccard": round(avg_jaccard, 4),
    "openai_average_latency_s": round(avg_openai_latency, 4),
    "v84_average_latency_s": round(avg_v84_latency, 4),
    "latency_difference_openai_minus_v84_s": round(
        avg_openai_latency - avg_v84_latency, 4
    ),
    "comparisons": comparisons,
}

JSON_OUT.write_text(
    json.dumps(json_output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)


md = []

md.append("# OpenAI vs V8.4 Competency Comparison\n")
md.append(
    "> This is a model-to-model agreement comparison on the same "
    "12 evaluation files. It is not an accuracy benchmark because "
    "no gold competency labels were provided.\n"
)

md.append("## Summary\n")
md.append(f"- Files compared: **{n}**")
md.append(
    f"- Exact competency-set agreement: "
    f"**{exact_matches}/{n} ({exact_matches/n:.1%})**"
)
md.append(
    f"- Difficulty agreement: "
    f"**{difficulty_matches}/{n} ({difficulty_matches/n:.1%})**"
)
md.append(
    f"- Average competency Jaccard similarity: **{avg_jaccard:.3f}**"
)
md.append(
    f"- OpenAI average latency: **{avg_openai_latency:.2f}s**"
)
md.append(
    f"- V8.4 average latency: **{avg_v84_latency:.2f}s**"
)
md.append(
    f"- Average latency difference "
    f"(OpenAI − V8.4): **{avg_openai_latency - avg_v84_latency:.2f}s**\n"
)

md.append("## File-by-file comparison\n")
md.append(
    "| File | OpenAI competencies | V8.4 competencies | "
    "Competency match | Jaccard | Difficulty match | "
    "OpenAI latency | V8.4 latency |"
)
md.append("|---|---|---|---|---:|---|---:|---:|")

for x in comparisons:
    md.append(
        f"| {x['file']} | "
        f"{'; '.join(x['openai_competencies'])} | "
        f"{'; '.join(x['v84_competencies'])} | "
        f"{'Yes' if x['competency_exact_match'] else 'No'} | "
        f"{x['competency_jaccard']:.3f} | "
        f"{'Yes' if x['difficulty_match'] else 'No'} | "
        f"{x['openai_latency_s']:.2f}s | "
        f"{x['v84_latency_s']:.2f}s |"
    )

md.append("\n## Differences\n")

for x in comparisons:
    if x["openai_only"] or x["v84_only"]:
        md.append(f"### {x['file']}")
        if x["openai_only"]:
            md.append(
                "- OpenAI only: "
                + ", ".join(x["openai_only"])
            )
        if x["v84_only"]:
            md.append(
                "- V8.4 only: "
                + ", ".join(x["v84_only"])
            )
        md.append(
            f"- Difficulty: OpenAI = "
            f"{x['openai_difficulty']}; "
            f"V8.4 = {x['v84_difficulty']}"
        )
        md.append("")

md.append("## Methodology\n")
md.append(
    "- Both systems were evaluated on the same 12 course-content files."
)
md.append(
    "- Competency predictions were compared as sets, ignoring ordering."
)
md.append(
    "- Jaccard similarity = intersection / union of competency sets."
)
md.append(
    "- Difficulty agreement checks exact equality of the difficulty label."
)
md.append(
    "- Latency is measured independently by each evaluation run."
)
md.append(
    "- No gold labels were available, so no accuracy, precision, "
    "recall, or F1 conclusions are made."
)

MD_OUT.write_text("\n".join(md), encoding="utf-8")

print("=" * 70)
print("OPENAI VS V8.4 COMPARISON")
print("=" * 70)
print(f"Files compared:              {n}")
print(
    f"Exact competency agreement:  "
    f"{exact_matches}/{n} ({exact_matches/n:.1%})"
)
print(
    f"Difficulty agreement:        "
    f"{difficulty_matches}/{n} ({difficulty_matches/n:.1%})"
)
print(f"Average Jaccard:              {avg_jaccard:.3f}")
print(f"OpenAI avg latency:           {avg_openai_latency:.2f}s")
print(f"V8.4 avg latency:             {avg_v84_latency:.2f}s")
print(
    f"Latency difference:            "
    f"{avg_openai_latency - avg_v84_latency:.2f}s"
)
print()
print(f"JSON: {JSON_OUT}")
print(f"Markdown: {MD_OUT}")
