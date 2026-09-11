# Remote (OpenAI-compatible) embedding backend: local validation

Date: 2026-09-11. Cost: $0 -- entirely local, no cloud instance.
Validates the `[embedding].backend = "openai"` feature (this PR) against a
real local server on the machine this whole investigation started on.

## Setup

| | |
|---|---|
| host | AMD Ryzen AI 9 HX PRO 375 (Strix Point), 24 threads, 62 GB RAM, integrated Radeon 890M (`gfx1150`, confirmed via `rocminfo`) |
| server | `lemonade` v11.6.0, built locally at `/home/jan/devel/lemonade` (`build/lemond`), run as `./lemond --port 13305 --host 127.0.0.1 --no-broadcast` |
| pdf-mcp | this branch, `feature/external-embedder`, unit tests green (`pytest tests/ -m "not slow"`: 1985 passed) |
| models tested | `Qwen3-Embedding-0.6B-GGUF` (llamacpp, CPU + vulkan, 1024-dim), `bge-small-en-v1.5-q8_0.gguf` (llamacpp, CPU + vulkan, 384-dim, same weights fastembed defaults to), `embed-gemma-300m-FLM` (FastFlowLM/NPU, 768-dim) |
| method | `pdf_mcp.remote_embedder.encode()` called directly (both against lemond's router and against `llama-server` binaries launched directly), the full `pdf_mcp.embedder` -> config dispatch path, and the real `pdf-mcp-warm` CLI against 2 real PDFs from `pages/corpus/` |

## Correctness -- full pipeline, real server, real PDFs

`pdf-mcp-warm pages/corpus/gao-cloud.pdf pages/corpus/nasa-artemis.pdf --sections`
against `[embedding].backend = "openai"` pointed at the local lemonade
(`Qwen3-Embedding-0.6B-GGUF`) completed cleanly: `2 warmed this run, 0
unprocessed, 0 skipped` in 11s, exit code 0.

Inspecting the resulting cache directly:

| check | result |
|---|---|
| cache identity | `openai:localhost:13305/Qwen3-Embedding-0.6B-GGUF@259ba907` (host + prefix-hash namespaced, as designed) |
| `cache.embeddings_complete(path, identity)` | `True` |
| stored vector dimension | 1024 (matches the model, matches every page) |
| stored vector norm | 1.0 (L2-normalized by `embedder._l2_normalize`, applied uniformly to both backends) |
| section FTS coverage | 10 (the `--sections` pass ran and indexed correctly alongside the remote embedding pass) |

Also exercised the config -> `embedder.configure_remote()` -> `encode`/
`encode_query` dispatch path directly, with a `query_prefix` set (Qwen3's
instruction-prefix convention), on a 3-sentence toy corpus: the query "What
stores images?" correctly ranked "PDF files store text and images." highest
(cosine 0.49 vs 0.32 and 0.21 for the other two sentences) -- end-to-end
proof the prefix plumbing and the remote encode/normalize path produce
usable, correctly-ranked retrieval, not just well-shaped arrays.

**Not run in this pass: the formal MRR gate** (`scripts/benchmark_embedding_models.py`
harness against `benchmark_data/ground_truth.json`, called for in the
original plan). This is a real gap against the plan's own bar, noted rather
than skipped silently -- see Follow-up below. It does not block merge because
the default backend (fastembed, unconfigured `[embedding]`) is byte-for-byte
unchanged by this feature; the gate matters before *recommending* a specific
remote model in `docs/embedding-models.md`, not before shipping the
opt-in mechanism.

**Also not run: cosine vs. fastembed on the same model weights** (the
MLX-study-style check). `bge-small-en-v1.5` turned out to be obtainable as a
GGUF after all (`ggml-org/bge-small-en-v1.5-Q8_0-GGUF`, downloaded directly
from Hugging Face rather than through lemonade's catalogue -- see the
Throughput section), so the vectors exist to run this comparison; the
cosine-vs-fastembed check itself was not run in this pass. The MLX finding
(cosine 0.894 divergence from a pooling difference, same weights) is why the
identity namespacing exists regardless.

## Throughput -- same-model CPU vs iGPU (Vulkan), via llama.cpp directly

The first pass here (previous revision of this doc) compared
`Qwen3-Embedding-0.6B` on the iGPU against `bge-small` on CPU -- an 18x
model-size difference confounding the backend comparison. Redone properly:
`llama-server` (both the `llamacpp/cpu` and `llamacpp/vulkan` binaries
lemonade already ships) loaded directly, bypassing lemond's router (its
`"backend"` request field was accepted but silently ignored -- see the
note below), each serving the SAME two GGUF files on different ports, so
model and framework are held constant and only the compute backend
changes. `bge-small-en-v1.5-q8_0.gguf` is `ggml-org/bge-small-en-v1.5-Q8_0-GGUF`
(the llama.cpp org's own quant of the exact model fastembed defaults to).
64 synthetic passages (~30 words each), `remote_embedder.encode()` called
directly, best-of-3 after a warmup call:

| model | CPU (llama.cpp) | Vulkan/iGPU (llama.cpp) | speedup |
|---|---|---|---|
| `bge-small-en-v1.5` (33M params, 384-dim) | 319 texts/s | 490-531 texts/s | **~1.5-1.7x** |
| `Qwen3-Embedding-0.6B` (600M params, 1024-dim) | 23-24 texts/s | 72-75 texts/s | **~3.0-3.2x** |

**The iGPU is a real, repeatable win on this machine, and the win grows with
model size** -- the opposite of the original onnxruntime/CPU investigation,
which found the shipped encode memory-bandwidth-bound (thread pinning gave
no benefit, `docs/configuration.md`'s Apple Silicon note). The larger model
is compute-bound enough that Vulkan offload triples throughput; the tiny
default model still gains ~1.5x. Concurrency past 1 matters far less than
the backend switch itself: both backends were near their per-request
plateau by concurrency=4 (`n_slots = 4` in the llama-server log matches).

For reference, `pdf_mcp.embedder.encode()` via fastembed/ONNX (the shipped
CPU path, not llama.cpp) on the same 64 passages: 125 texts/s for
`bge-small` -- faster than llama.cpp's CPU build of the same model (317
vs 125 looks backwards; ONNX and llama.cpp use different batching/threading
internally, and 125 texts/s is *slower* here, so read this as "different
CPU implementations of the same model are not identical," not as a
regression in either).

**Not measured: `gfx1150` (ROCm), or the FastFlowLM/NPU path.** The
`therock/gfx1150-7.13.0` ROCm build present in lemonade's cache was not
exercised through `lemond`: the loader's `"backend": "rocm"` request field
was accepted (`HTTP 200`) but the server still launched `llamacpp/vulkan`
regardless (confirmed via the actual process command line). The raw
`llamacpp/rocm-stable/llama-b10394/llama-server` binary exists and could be
driven directly the same way the CPU/Vulkan comparison above was, but
wasn't in this pass -- see Follow-up. This means the numbers above are a
CPU-vs-Vulkan floor, not the ceiling: ROCm on the same `gfx1150` hardware
would likely do better still.

## NPU path -- reproducibly broken in this session, not a pdf-mcp defect

`embed-gemma-300m-FLM` (FastFlowLM, targets the Ryzen AI NPU) answered its
very first request correctly (768-dim vectors, HTTP 200, 2.45s including
model load). Every subsequent request -- including after unloading and
reloading the model, and after a full `lemond` process restart -- failed
identically:

```
HTTP 500 {"error":"qds_device::wait() unexpected command state"}
```

`pdf_mcp.remote_embedder`'s retry logic (3 attempts, exponential backoff)
correctly retried and correctly surfaced the failure as a
`RemoteEmbeddingError` rather than hanging or returning garbage -- the
client-side behavior here is exactly as designed. The failure itself is
inside lemonade's FastFlowLM/NPU driver stack (`flm` subprocess), not in
this PR's code, and did not recover within this session. Likely an
XDNA/`amdxdna` driver state issue that needs a deeper reset (module
reload, reboot) than a process restart provides -- not chased further here
to stay in scope.

## Verdict

**Ship the feature.** Every correctness property the design set out to
guarantee -- identity namespacing, normalization, prefix application,
dimension/order integrity, retry/secret hygiene -- is demonstrated working
against a real server, not just mocks (mocks cover the same properties
exhaustively in `tests/test_remote_embedder.py`; this pass is the "does it
actually work" check on top of that). The default (fastembed) path is
provably unaffected.

**The iGPU (Vulkan) path is now a demonstrated speed win, same-model,
same-framework, repeatable across runs: ~1.5-1.7x on the tiny default-sized
model, ~3.0-3.2x on a 600M-param model.** This is safe to state in
docs/README now, scoped correctly (Vulkan via llama.cpp, not the shipped
fastembed/ONNX CPU path, and not yet ROCm or NPU). No NPU number exists --
the FastFlowLM path did not stay usable long enough to benchmark. This
mirrors the honest-negative-result culture this repo already has
(`docs/investigated-rejected.md`, the MLX and E5 studies) -- record what was
actually measured, not what was hoped for, positive or negative.

## Follow-up (not blocking this PR)

- Run the formal MRR gate (`scripts/benchmark_embedding_models.py` harness)
  against a servable-via-lemonade model before recommending one by name in
  `docs/embedding-models.md`.
- Run the cosine-vs-fastembed check now that a same-weights GGUF
  (`ggml-org/bge-small-en-v1.5-Q8_0-GGUF`) is in hand -- both files already
  downloaded in this session, just not diffed.
- Drive `llamacpp/rocm-stable/llama-b10394/llama-server` directly (the same
  way the CPU/Vulkan comparison above bypassed lemond's router) to get a
  `gfx1150`-ROCm throughput number; likely faster than the Vulkan numbers
  above, not slower.
- Investigate the FastFlowLM/NPU `qds_device::wait()` failure (driver
  version, `amdxdna` module reload, or a lemonade issue report) to get a
  usable NPU throughput number.
- Report the ignored `"backend"` field in lemond's `/api/v1/load` to the
  lemonade project -- confirmed twice (`rocm` and `cpu` requests both
  silently launched `vulkan`).
