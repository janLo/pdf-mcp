# German retrieval-quality benchmark (BGB) — local and remote models

> Part of the [embedding evaluation summary](embedding_evaluation_summary.md).
> Answers [jztan/pdf-mcp#46](https://github.com/jztan/pdf-mcp/issues/46): the
> original numbers (bge-small MRR 0.075 on natural-language German queries,
> vs 0.266 for a remote `bge-m3`) came from a private German commentary that
> can't be shared. This is the same style of benchmark on a document anyone
> can download, extended to also cover the remote-endpoint side of the issue.

## A note on process

The first version of this benchmark (n=90 sampled norms, 29 usable
natural-language pairs) had two real methodology bugs, both caught before
anything was posted: an independent review found that (a) roughly 14% of
the sampled cross-references actually cited a *different* statute (the BGB
cites other codes constantly — "§ 109 der Zivilprozessordnung" is not BGB
§ 109), mislabeling those scenarios, and (b) the natural-language query text
is lifted verbatim from a *citing* page, so a model that retrieves that
citing page back was being scored as a miss even though finding the
sentence's own source is not a wrong answer. Both depressed every model's
MRR by an unknown, model-dependent amount, and (b) especially could have
distorted the *ranking* between models, not just the absolute numbers. Both
are fixed in `scripts/gen_german_ground_truth.py` (see its git history for
detail); the numbers below are from the corrected corpus, at 300 sampled
norms this time (a bigger sample was cheap once available, per
`docs/contributing.md`'s "Quality loop" — small-sample benchmarks overstate
the gap). A second review pass found and fixed four more correctness bugs in
the generator and the harness (word-boundary citation matching, a
sentence-split truncation on legal abbreviations, an extent-ordering bug on
duplicate norm numbers, and two silent-failure paths in the harness) — see
the same commit history.

## Corpus and method

Ground truth is derived mechanically from
[gesetze-im-internet.de's BGB.pdf](https://www.gesetze-im-internet.de/bgb/BGB.pdf)
(490 pages, born-digital) — **no manual annotation, no LLM**. Method, in
`scripts/gen_german_ground_truth.py`:

1. The PDF outline lists every norm as `§ N Rubrik` with a resolvable page
   (2,510 such entries after dropping `(weggefallen)`/repealed norms and
   `§§ N bis M` ranges). Each norm's own citation and rubric are verified to
   actually occur on its anchor page (word-boundary matched) before it's
   trusted.
2. The body text cites other norms constantly (~10 `§`-citations per page).
   For each sampled norm, one citing page elsewhere in the document supplies
   a natural-language query: the sentence leading up to the citation, with
   every `§`, `Absatz`/`Abs.`/`Satz`/`Nr.` token and bare digit stripped out,
   wrapped in a fixed German question frame (`"Was gilt für …?"` /
   `"Wo ist … geregelt?"`). This carries no verbatim citation for keyword
   search to win on for free. Citations to a *different* statute are
   filtered out (59 of them, across the whole document). The referrer page
   itself counts as relevant alongside the cited norm's own page(s) — the
   query text came from there, so retrieving it is a correct answer, not a
   miss.
3. A second, deliberately easy arm per norm uses the rubric verbatim as a
   keyword-style query — the **control** that proves the corpus itself
   isn't broken, independent of embedding model.

300 norms sampled (`random.Random(20260913)`, stratified across the BGB's
five *Bücher*, `§ 611a` force-included per the issue discussion). 119
produced a usable natural-language pair after the anti-triviality filters (a
citation too close to its own target extent, an empty clause, a query that
mostly restated the rubric, or a citation to a different statute are all
rejected rather than silently re-rolled — see skip counts in
`benchmark_data/german_ground_truth_provenance.json`); every sampled norm
gets the keyword-control arm regardless (300 scenarios), so the committed
ground truth carries **419 scenarios** total.

Reproduce:
```
python scripts/gen_german_ground_truth.py           # regenerates the ground truth
python scripts/benchmark_embedding_models.py \
  --ground-truth benchmark_data/german_ground_truth.json \
  --mode semantic --arms semantic_xref \
  --models BAAI/bge-small-en-v1.5,<candidate>
```

## Keyword control arm (sanity check)

```
python scripts/benchmark_embedding_models.py \
  --ground-truth benchmark_data/german_ground_truth.json \
  --mode keyword --arms keyword_control
```

| Mode | MRR |
|------|-----|
| keyword, rubric queries (300 scenarios) | **0.850** |

High and model-independent, as expected — this confirms the ground truth
and German text extraction are sound. Any gap seen below on the semantic
arm is a *model* limitation, not a corpus problem.

## Semantic arm — 119 scenarios, `mode=semantic`, `k=5`

All runs were sequential (never overlapping) on an otherwise idle machine,
so the latency numbers here are far less noisy than an earlier draft's
(which ran under concurrent background load and showed ~2x self-spread on
identical repeat runs). bge-small's own MRR was **perfectly stable at
0.383** across all five runs below, each against a freshly re-embedded
cache — a good reproducibility signal.

### Local (fastembed, CPU) — reproducible by anyone with this repo

| Model | MRR | Δ vs bge-small | p50 query | Cold embed (490p) | Dim |
|-------|-----|-----------------|-----------|--------------------|-----|
| `BAAI/bge-small-en-v1.5` *(default, English)* | 0.383 | — | 48 ms | 140 s | 384 |
| `intfloat/multilingual-e5-large` | **0.632** | **+0.249** | 87 ms | 807 s | 1024 |
| `jinaai/jina-embeddings-v2-base-de` | **0.565** | **+0.182** | 83 ms | 356 s | 768 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0.340 | −0.043 | 47 ms | 49 s | 384 |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 0.313 | −0.070 | 58 ms | 220 s | 768 |

**`jina-embeddings-v2-base-de` — the model the issue specifically asked
about — DOES work locally, contrary to an earlier draft of this document.**
It fails to load under fastembed's hardcoded
`onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL` (a `graph_utils.cc`
assertion in the `SimplifiedLayerNormFusion` pass, reproducible standalone
with the raw ONNX file), but loads and embeds correctly at
`ORT_ENABLE_EXTENDED` (one level down) or lower. `scripts/
benchmark_embedding_models.py --patch-onnx-graph-opt` downgrades that one
level, process-wide, for the duration of the benchmark run only (never
touches `src/pdf_mcp/embedder.py`), and every number in the table above for
this model was produced with it. Shipping this as a permanent fix for the
production path would need `embedder.py` to accept a per-model session-
options override — out of scope here, tracked as a fast-follow if `jina-de`
is adopted.

**Both `multilingual-e5-large` and the patched `jina-de` clear the existing
+0.05 MRR-lift gate**, but neither clears the existing 1.5x latency gate on
p50 query time (e5-large 1.81x, jina-de 1.73x) — though note the gate is
tuned for the *English* arxiv corpus's decision, not written with German in
mind, so whether it should bind here is a real open question, not a
foregone conclusion. Both run raw, with **no** `query:`/`passage:` prefix —
matching pdf-mcp's actual production path today (no prefix mechanism
exists in `embedder.py`) rather than either model's documented-optimal
usage, so these numbers likely understate what they could do with prefix
support, at the cost of the extra machinery
`benchmark_data/e5_prefix_results.md` already found net-negative for
English.

**`MiniLM` and `mpnet` both score modestly below baseline** (−0.043,
−0.070) — with 119 scenarios this is a real signal, not the single- or
double-scenario noise a smaller sample would have produced, but the
absolute gap is small. Multilingual training does not automatically help
here, matching this repo's own English large-model screen's conclusion
that bigger/broader doesn't automatically transfer to this retrieval
regime. `MiniLM` is also markedly cheaper than the default (49s vs 140s
cold embed) for a modest quality cost — worth keeping in mind as a
budget option even though it doesn't pass the lift gate.

### Remote (OpenAI-compatible `/v1/embeddings`, via `lemond` on Vulkan)

The remote backend merged in #47 only accepts an endpoint serving a
bge-small-compatible model, and the model-choice extension that would lift
that restriction is not merged, so this arm used a standalone scorer
(`scripts/gen_german_ground_truth.py`'s sibling logic, not committed —
mirrors `_compute_metrics`/cosine-on-L2-normalized-vectors exactly) talking
directly to `lemond`'s OpenAI-compatible endpoint. Pooling set per model via
`lemonade config set llamacpp.args="--pooling {cls,last}"` (bge-m3: CLS;
Qwen3-Embedding: last-token, per each model's documented pooling).

| Model | MRR | Δ vs bge-small | Cold embed (490p) | Dim |
|-------|-----|-----------------|---------------------|-----|
| `bge-m3` (Q8_0 GGUF) | 0.509 | +0.126 | 94 s | 1024 |
| `Qwen3-Embedding-0.6B` (Q8_0 GGUF) | 0.451 | +0.068 | 265 s | 1024 |

**The per-query timing for this arm is not directly comparable to the local
table above** — the standalone scorer batches all queries into a handful of
HTTP requests and divides total wall time by query count, rather than
timing individual round-trips the way `run_latency_probe` does for the
local models. Cold-embed time (a single large batch job either way) *is*
comparable, and there both remote models embed 490 pages 1.5-3.7x faster
than the local models scored above them in MRR (94s/265s vs e5-large's
807s, jina-de's 356s) — the throughput case for a remote/accelerated
backend the original private-corpus benchmark made still holds.

**The genuinely surprising result: on this corpus, with the leak fixed,
neither remote model is the best option.** Both local `e5-large` (0.632)
and local `jina-de` (0.565) outperform both `bge-m3` (0.509) and
`Qwen3-Embedding-0.6B` (0.451) on MRR. That inverts the headline framing of
the original issue (private corpus: remote `bge-m3` 0.266 handily beat
local `bge-small` 0.075). Plausible reasons, none confirmed here: this
corpus's domain (statute text, short factual cross-references) differs
substantially from the original commentary; the remote arm ran with no
prefix support either, which may hurt `bge-m3`/`Qwen3-Embedding` more than
it hurts `e5-large`/`jina-de` (uncertain, not tested); and 119 scenarios,
while much better-powered than the original 29, is still a single corpus,
single domain — the standard caveat this repo's own `docs/
contributing.md` "Quality loop" raises about any single-sample benchmark.

## What this means for the issue

- **The corpus is fine; bge-small's German gap is real.** Keyword search on
  the rubric arm (0.850) is high and model-independent; bge-small's
  semantic MRR (0.383) sits well below every other model tested except the
  two English-trained multilingual sentence-transformers.
- **`jina-embeddings-v2-base-de` (ask #2) works locally**, with a one-line,
  documented, process-scoped workaround for an `onnxruntime`
  graph-optimization incompatibility, and it meaningfully closes the gap
  (+0.182 MRR). It does not currently clear the existing MRR-lift *and*
  latency combined gate, but it is now at least a real, testable candidate,
  which it was not in an earlier draft of this benchmark.
- **A local model can match or beat what the private-corpus numbers
  suggested only a remote endpoint could do.** `e5-large` and `jina-de`
  both outperform both remote models tested here. This doesn't mean the
  remote-backend work (#47/#48) is unnecessary — the throughput numbers
  above still favor remote/accelerated serving, and this is one corpus in
  one domain — but it does mean "German needs a remote endpoint" is not
  established by this benchmark; if anything it points the other way for
  this corpus.
- No model tested (local or remote) is a clean, gate-passing drop-in
  default replacement for bge-small. Whether German support should key off
  a *different* default model, a user-configurable one via the existing
  BYOM path (already possible today for `jina-de` and `e5-large` with the
  `--patch-onnx-graph-opt`-equivalent fix ported into `embedder.py`), or
  the remote backend, is a real design decision this benchmark informs but
  doesn't settle on its own.

## English regression check

The same harness, unmodified corpus (`benchmark_data/ground_truth.json`),
`BAAI/bge-small-en-v1.5` only — confirms the `--models`/`--mode`/`--arms`
additions to `scripts/benchmark_embedding_models.py` don't change the
existing English default's measured quality.

| Model | MRR | p50 latency |
|-------|-----|-------------|
| `BAAI/bge-small-en-v1.5` *(default)* | 0.726 | 61.6 ms |

**Running this exposed a real, pre-existing bug, unrelated to anything in
this change**: `ground_truth.json` on `develop` carries two PDF entries
("bert", "resnet") with an empty `"scenarios": {}` — added in a prior
commit reserving them for a corpus this harness doesn't yet consume — and
`run_model`'s per-PDF warm-up loop called `next(iter(pdf["scenarios"]))`
unconditionally, crashing with `StopIteration`. Verified against
`origin/develop`'s own unmodified `scripts/benchmark_embedding_models.py`
and its own committed `ground_truth.json` in an isolated worktree — this
crash predates this branch and would hit anyone running the harness today
with no arguments. Fixed alongside (skip a PDF with no scenarios in the
warm-up loop; the latency probe now picks the first PDF that actually has
a warmed query), with regression tests.

The MRR above (0.726) differs from `docs/embedding-models.md`'s committed
table (0.806, dated 2026-05-09) — expected drift since that run, not
something this change touches.
