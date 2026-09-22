# Infrastructure benchmark (W3D5) - Lama

Benchmark of Qwen2.5-7B-Instruct-AWQ served with vLLM on a single Tesla T4.

## Files

| File | Purpose |
|---|---|
| `w3d5_benchmark.ipynb` | Colab notebook. Run the cells in order, once each. Part A = lab, Part B = real course content |
| `bench.py` | The lab's benchmark harness, unmodified |
| `prompts.txt` | The lab's 20 fixed prompts |
| `verify_cell.py` | The lab's green check |
| `model-lock.md` | Model, flags and versions used |
| `requirements.txt` | Pinned package versions |
| `Infrastructure_Benchmark_Report.md` | The report |

Outputs produced by the notebook (add them here after running): `bench_report.json`, `knee.json`,
`capacity-note.md`, `bench_report_real.json`, `prompts_real.txt`.

## Two runs, on purpose

1. **Lab run** - `prompts.txt`, prefix cache on. Produces the knee at the SLO and the capacity note.
2. **Real-content run** - the 12 course files, prefix cache off, `--max-tokens 400`. Produces the numbers used for cost.

Real course files are new on every request, so a run that reuses cached prompts flatters both latency and cost.
