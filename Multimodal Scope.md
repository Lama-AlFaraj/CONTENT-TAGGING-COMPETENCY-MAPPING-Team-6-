## Multimodal Scope

The final V8.4 pipeline was designed and evaluated primarily for text-based learning content. However, the assigned use case can contain visually encoded information such as diagrams, charts, and images embedded within presentation slides.

Some evaluation content contains visual information that may not be fully represented in extracted text. A multimodal feasibility evaluation was therefore considered as a potential extension to assess whether visual information could improve concept and competency identification. However, this evaluation was not conducted and is not part of the validated V8.4 benchmark.

### Multimodal Feasibility Test

A potential vision-language evaluation would provide selected slide images together with their available textual content to a vision-language model. The purpose would be to determine whether visual information can recover concepts that a text-only extraction pipeline may miss.

This was considered as a **future feasibility evaluation**, not as part of the implemented V8.4 production pipeline.

### Current Scope

| Approach              | Input                | Role                                     |
| --------------------- | -------------------- | ---------------------------------------- |
| V8.4 Text Pipeline    | Extracted text       | Main validated approach                  |
| E5 Retrieval          | Text representations | Semantic retrieval component             |
| Qwen LLM              | Text/context         | Tagging and reranking                    |
| Vision-Language Model | Slide images + text  | Considered future feasibility evaluation |

### Scope Boundary

The final deployed V8.4 system remains text-based. Full multimodal production support, including systematic image extraction, visual embedding/retrieval, multimodal reranking, and large-scale multimodal benchmarking, is outside the validated implementation scope.

Multimodal processing is therefore documented as a potential future extension rather than a completed experiment or validated component of the final system.
