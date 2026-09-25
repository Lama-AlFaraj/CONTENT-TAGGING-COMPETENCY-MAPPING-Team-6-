# Cost Analysis

## OpenAI Baseline

The OpenAI baseline was evaluated on the same 12-file evaluation dataset used for the competency-mapping comparison.

### Measured Usage

| Metric              |   Value |
| ------------------- | ------: |
| Evaluation files    |      12 |
| API calls           |      80 |
| Input tokens        | 111,465 |
| Cached input tokens |       0 |
| Output tokens       |  12,622 |
| Total tokens        | 124,087 |

### API Cost

Using the applicable OpenAI API pricing for the evaluated model:

| Component     |       Usage |       Rate |        Cost |
| ------------- | ----------: | ---------: | ----------: |
| Input tokens  |     111,465 | $0.20 / 1M |     $0.0223 |
| Output tokens |      12,622 | $1.20 / 1M |     $0.0151 |
| **Total**     | **124,087** |          — | **$0.0374** |

The measured OpenAI evaluation therefore cost approximately **$0.0374 for the 12-file benchmark**, or approximately **$0.00312 per evaluation file**.

## Self-Hosted V8.4

V8.4 uses a self-hosted Qwen2.5-7B-Instruct-AWQ model served through vLLM on an NVIDIA RTX A6000 48 GB GPU.

Unlike the OpenAI baseline, the self-hosted model does not incur a per-token API charge. Its cost depends on the infrastructure used to run the GPU, including GPU runtime, compute capacity, and associated infrastructure.

An exact dollar cost is not claimed here because the project did not record an actual GPU billing rate for the BeamData infrastructure.

## Cost Comparison

The OpenAI benchmark provides a measured API cost of approximately **$0.0374 for 12 evaluation files**.

The V8.4 deployment provides a self-hosted inference path where cost is determined by infrastructure utilization rather than API token pricing. A direct dollar comparison requires an approved or actual BeamData GPU-hour cost.

## Evidence

The measured token usage is recorded in:

`Infrastructure_Benchmark/results/openai_cost_evaluation.json`

The existing OpenAI model comparison is documented separately in the benchmark comparison artifacts.
