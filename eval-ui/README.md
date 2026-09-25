# Evaluation / Testing UI

A single self-contained page (`index.html`) that implements the bootcamp's Deliverable 6: provide content, run it against one or more model endpoints, and inspect/compare the structured outputs. No build step — just open the file in a browser, or drop it into `ai-hub/` and serve it as a static file.

## Setup

1. Open `index.html` in a browser (double-click, or serve it from anywhere — it makes no network calls of its own besides the endpoints you configure).
2. Under **1. Endpoints**, point at your running BeamData API, e.g. `http://127.0.0.1:8002` if you're port-forwarding the Kubernetes service the way the README describes. Add a second row to compare a second model/backend (e.g. an OpenAI-comparison shim, or a second vLLM instance serving a different model) side by side.
3. Upload a file (`.pptx`, `.ipynb`, `.xlsx`, `.md`, `.txt`, `.csv`) or paste text.
4. Click **Run**. Each endpoint gets called with the same content; results and a comparison table render below.

Endpoint list is saved in your browser's local storage so you don't have to retype URLs every session.

## CORS — the one real gotcha

Browsers block cross-origin requests unless the server explicitly allows them. `Model/v8/api.py` does not currently set CORS headers, so opening this page from `file://` or from a different port than the API will fail with a network error (the UI will surface this as an error card, not a silent failure). Fix it by adding to `Model/v8/api.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your actual origin(s) before sharing this beyond local testing
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

Restart the API after adding this. If you're testing against a raw vLLM OpenAI-compatible endpoint instead of the FastAPI wrapper, note the UI expects the BeamData `/predict` contract (multipart `file` field, JSON response with `predicted_tags` / `proposed_competencies` / `difficulty_level` / `confidence` / `notes`) — it isn't wired for vLLM's own `/v1/chat/completions` shape directly.

## What it does not do

- It doesn't run the model itself — it's a thin client. All actual inference happens on whatever endpoint(s) you point it at.
- It doesn't score outputs against gold labels — that's still `Model/v8/evaluate_v8.py`. This tool is for eyeballing/comparing outputs, per the mentor's "evaluation/demo focus, not frontend development" scoping note.
