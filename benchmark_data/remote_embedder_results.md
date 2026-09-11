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
| models tested | `Qwen3-Embedding-0.6B-GGUF` (llamacpp/vulkan, 1024-dim) and `embed-gemma-300m-FLM` (FastFlowLM/NPU, 768-dim) |
| method | `pdf_mcp.remote_embedder.encode()` called directly, and the full `pdf_mcp.embedder` -> config dispatch path, and the real `pdf-mcp-warm` CLI against 2 real PDFs from `pages/corpus/` |

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
MLX-study-style check). lemonade's llamacpp backend serves GGUF checkpoints;
`bge-small-en-v1.5` is not in its catalogue as a GGUF, so a true
same-weights comparison needs either a GGUF conversion of bge-small or a
server that can serve it directly -- out of scope for this pass. The
MLX finding (cosine 0.894 divergence from a pooling difference, same
weights) is why the identity namespacing exists regardless of whether this
specific pair could be measured.

## Throughput -- vulkan (iGPU) vs fastembed CPU, at varying concurrency

64 synthetic passages (~30 words each), `remote_embedder.encode()` called
directly to isolate the network+backend cost from pdf-mcp's own batching:

| condition | time | throughput |
|---|---|---|
| remote, `Qwen3-Embedding-0.6B-GGUF` (vulkan/iGPU), concurrency=1 | 1.20s | 53.2 texts/s |
| remote, concurrency=2 | 0.97s | 66.1 texts/s |
| remote, concurrency=4 | 0.90s | 70.9 texts/s |
| remote, concurrency=8 | 0.86s | 74.6 texts/s |
| fastembed CPU, `bge-small-en-v1.5` | 0.37s | 170.8 texts/s |

**Read this carefully -- it is not an iGPU-vs-CPU verdict.** `Qwen3-Embedding-0.6B`
is an 18x larger model (600M params vs bge-small's 33M) than the CPU
baseline; the comparison is confounded by model size, not backend. It does
show two useful things: concurrency helps (53 -> 75 texts/s, 1.4x, plateauing
by concurrency=4, consistent with 4 llama.cpp server slots — see `n_slots = 4`
in the server log), and that a much larger, likely higher-quality model
stays in the same order of magnitude as the tiny CPU default once network
and iGPU are both in the loop -- i.e. the offload is fast enough to be
useful, even though this pair doesn't prove the iGPU beats the CPU for
one fixed model. A fair same-model CPU-vs-vulkan-vs-`gfx1150`(ROCm)
comparison needs a GGUF small enough for lemonade's CPU backend, which is
listed as "installable" but was not installed in this pass (see Follow-up).

The `therock/gfx1150-7.13.0` ROCm build present in lemonade's cache was
not exercised: the loader's `"backend": "rocm"` request field was accepted
(`HTTP 200`) but the server still launched `llamacpp/vulkan` regardless
(confirmed via the actual process command line) -- selecting it needs
either a lemonade config/env knob not found in this session, or a fresh
model catalogue entry pinned to that recipe. Left as follow-up rather than
worked around by guessing.

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

**Do not yet claim an NPU or iGPU speed win in docs/README.** The iGPU
(vulkan) path works and is fast enough to be practically useful, but no
same-model comparison was obtained to say it beats CPU for one fixed model,
and the NPU path did not stay usable long enough to benchmark at all. This
mirrors the honest-negative-result culture this repo already has
(`docs/investigated-rejected.md`, the MLX and E5 studies) -- record what was
actually measured, not what was hoped for.

## Follow-up (not blocking this PR)

- Run the formal MRR gate (`scripts/benchmark_embedding_models.py` harness)
  against a servable-via-lemonade model before recommending one by name in
  `docs/embedding-models.md`.
- Install lemonade's `llamacpp:cpu` backend and re-run the throughput table
  with the *same* model (e.g. `Qwen3-Embedding-0.6B-GGUF`) on CPU vs vulkan
  vs (if selectable) `gfx1150`-ROCm, to get an apples-to-apples GPU-offload
  number.
- Work out how to pin lemonade to the `therock/gfx1150` ROCm backend rather
  than its vulkan default, and re-run.
- Investigate the FastFlowLM/NPU `qds_device::wait()` failure (driver
  version, `amdxdna` module reload, or a lemonade issue report) to get a
  usable NPU throughput number.
