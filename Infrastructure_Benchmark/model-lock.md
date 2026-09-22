# Model lock

> Reconstructed from the configuration that was actually run. Check it against the team's
> original model-lock from day 4 and correct any line that differs.

| Item | Value |
|---|---|
| Model id | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| Quantization | AWQ 4-bit (`--quantization awq`) |
| Serving engine | vLLM 0.29.0 (OpenAI-compatible server) |
| GPU | 1x Tesla T4, 15,360 MB VRAM (Google Colab) |
| dtype | float16 (`--dtype half`; T4 has no bfloat16) |
| Max model length | 16384 (8192 rejected the two longest course files) |
| GPU memory utilization | 0.85 |
| Attention backend | TRITON_ATTN (chosen automatically; FlashAttention 2 needs compute capability >= 8) |
| Environment variable | `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer sampler unsupported on T4) |
| Prefix caching | On for the lab run (vLLM default); off for the real-content run |
| Python / torch | 3.13 / 2.13.0 (cu130) |

## Launch command

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --dtype half \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --port 8000
# real-content run: add --no-enable-prefix-caching
```

## Known pitfalls

- Wait for `/v1/models` to answer before benchmarking; startup takes 2 to 5 minutes.
- The model id passed to the harness must end in `-AWQ`, or every request returns model-not-found.
- Uninstall `torchaudio` after installing vLLM; keep `torchvision` (vLLM imports it).
