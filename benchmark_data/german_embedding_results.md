# German retrieval-quality benchmark (BGB) — local fastembed models

> Part of the [embedding evaluation summary](embedding_evaluation_summary.md).
> Answers [jztan/pdf-mcp#46](https://github.com/jztan/pdf-mcp/issues/46): the
> original numbers (bge-small MRR 0.075 on natural-language German queries,
> vs 0.266 for a remote `bge-m3`) came from a private German commentary that
> can't be shared. This is the same style of benchmark on a document anyone
> can download.

## Corpus and method

Ground truth is derived mechanically from
[gesetze-im-internet.de's BGB.pdf](https://www.gesetze-im-internet.de/bgb/BGB.pdf)
(490 pages, born-digital) — **no manual annotation, no LLM**. Method, in
`scripts/gen_german_ground_truth.py`:

1. The PDF outline lists every norm as `§ N Rubrik` with a resolvable page
   (2,510 such entries after dropping `(weggefallen)`/repealed norms and
   `§§ N bis M` ranges). Each norm's own citation and rubric are verified to
   actually occur on its anchor page before it's trusted.
2. The body text cites other norms constantly (~10 `§`-citations per page).
   For each sampled norm, one citing page elsewhere in the document supplies
   a natural-language query: the sentence leading up to the citation, with
   every `§`, `Absatz`/`Abs.`/`Satz`/`Nr.` token and bare digit stripped out,
   wrapped in a fixed German question frame (`"Was gilt für …?"` /
   `"Wo ist … geregelt?"`). This carries no verbatim citation for keyword
   search to win on for free.
3. A second, deliberately easy arm per norm uses the rubric verbatim as a
   keyword-style query — the **control** that proves the corpus itself
   isn't broken, independent of embedding model.

90 norms sampled (`random.Random(20260913)`, stratified across the BGB's five
*Bücher*, `§ 611a` force-included per the issue discussion). 29 produced a
usable natural-language pair after the anti-triviality filters (a citation
too close to its own target page, an empty clause, or a query that mostly
restated the rubric are all rejected rather than silently re-rolled — see
skip counts in `benchmark_data/german_ground_truth_provenance.json`); every
sampled norm gets the keyword-control arm regardless (90 scenarios), so the
committed ground truth carries **119 scenarios** total.

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
| keyword, rubric queries | **0.812** |

High and model-independent, as expected — this confirms the ground truth
and German text extraction are sound. Any gap seen below on the semantic
arm is a *model* limitation, not a corpus problem.

## Semantic arm — local fastembed models (CPU)

29 `semantic_xref` scenarios, `mode=semantic`, `k=5`. All models runnable
with no external server — every row here is reproducible by anyone with
this repo and a CPU.

| Model | MRR | Δ vs bge-small | Dim | Note |
|-------|-----|-----------------|-----|------|
| `BAAI/bge-small-en-v1.5` *(default, English)* | 0.172 | — | 384 | baseline |
| `intfloat/multilingual-e5-large` | **0.274** | **+0.102** | 1024 | raw, no prefix (see below) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0.143 | −0.029 | 384 | worse than baseline |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 0.123 | −0.049 | 768 | worse than baseline |
| `jinaai/jina-embeddings-v2-base-de` | — | — | 768 | **fails to load** (see below) |

**`jina-embeddings-v2-base-de` cannot be benchmarked as shipped.** This is
the model the issue specifically asked to add. It fails at `onnxruntime`
session initialization, reproducible standalone (not a harness artifact):

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL :
Exception during initialization: .../graph_utils.cc:29 ... itr != node_args.end()
was false. Attempting to get index by a name which does not exist:
InsertedPrecisionFreeCast_/encoder/layer.11/attention/output/LayerNorm/
Constant_output_0 for node: /embeddings/LayerNorm/Mul/SimplifiedLayerNormFusion/
```

An ONNX graph-fusion pass (`SimplifiedLayerNormFusion`) chokes on this
model's exported graph under the pinned `onnxruntime==1.24.4`. Same failure
class `docs/embedding-models.md` already documents for `nomic-embed-text-v1.5`
(segfault) and `gte-base` (`ValueError`) — a tooling incompatibility, not a
quality verdict. **So the direct answer to "would jina-de close the gap
locally?" is: it can't be tested at all on this stack today.**

**`multilingual-e5-large` is the one local model that moves the needle**
(+0.102 MRR, +59% relative over baseline) — but two real costs come with it:
cold-embedding all 490 BGB pages took **~17 minutes** on this machine (vs
~3-6 minutes for bge-small under the same concurrent load), and warm p50
query latency was measured at ~1.7x the baseline's, which is just over this
harness's existing 1.5x latency gate
(`scripts/benchmark_embedding_models.py`'s `compute_verdict`). It was run
**raw, with no `query:`/`passage:` prefix** — matching pdf-mcp's actual
production path today (`embedder.py` has no prefix mechanism at all) rather
than e5's documented-optimal usage, so this number likely understates what
e5-large could do with prefix support, at the cost of the extra machinery
`benchmark_data/e5_prefix_results.md` already found net-negative on the
English corpus.

*(Latency/embed-time numbers above were measured with other background jobs
competing for CPU on this machine and vary run-to-run by roughly 2x; treat
them as directional, not precise. MRR is deterministic and unaffected.)*

## What this means for the issue

- **Neither local multilingual model fully closes the gap** the original
  private-corpus numbers showed for a remote `bge-m3`/`Qwen3-Embedding-0.6B`
  endpoint (MRR 0.24-0.27 there). `multilingual-e5-large` gets into the same
  numeric range (0.274) on this different corpus, but doesn't clear this
  harness's own latency gate raw, and the two smaller multilingual models
  actively regress vs. the English default.
- **The specific model the issue asked about doesn't run** on this repo's
  pinned dependency versions. Adding it to `docs/embedding-models.md`'s
  validated list isn't possible until that's fixed upstream (in `fastembed`,
  `onnxruntime`, or the jina ONNX export itself).
- German does need *something* better than the default — the keyword-control
  arm proves the corpus is fine and bge-small's semantic MRR (0.172) is a
  real gap, not a benchmark artifact — but **no CPU-only local model tested
  here is a clean drop-in**. This is evidence for, not against, the remote
  backend + confidence-threshold registry discussion on this issue: the
  local path alone (as tested) doesn't make the remote backend unnecessary
  for German.

## English regression check

The same harness, unmodified corpus (`benchmark_data/ground_truth.json`),
`BAAI/bge-small-en-v1.5` only — confirms the `--models`/`--mode`/`--arms`
additions to `scripts/benchmark_embedding_models.py` don't change the
existing English default's measured quality.

| Model | MRR | p50 latency |
|-------|-----|-------------|
| `BAAI/bge-small-en-v1.5` *(default)* | 0.726 | 63.5 ms |

**Running this exposed a real, pre-existing bug, unrelated to anything in
this change**: `ground_truth.json` on `develop` now carries two PDF entries
("bert", "resnet") with an empty `"scenarios": {}` — added in a prior commit
reserving them for a corpus this harness doesn't yet consume — and
`run_model`'s per-PDF warm-up loop called `next(iter(pdf["scenarios"]))`
unconditionally, crashing with `StopIteration` on the first such entry.
Verified against `origin/develop`'s own unmodified
`scripts/benchmark_embedding_models.py` and its own committed
`ground_truth.json` in an isolated worktree — this crash predates this PR
and would hit anyone running the harness today with no arguments. Fixed
alongside (skip a PDF with no scenarios in the warm-up loop; the latency
probe now picks the first PDF that actually has a warmed query instead of
`next(iter(gt["pdfs"]))`), with regression tests
(`TestRunModel::test_tolerates_pdf_with_no_scenarios_yet` and
`test_empty_scenarios_pdf_ordered_first_still_works`).

The MRR above (0.726) differs from `docs/embedding-models.md`'s committed
table (0.806, dated 2026-05-09) — expected drift since that run, not
something this change touches; the two other silent-failure classes this
script already handles (a per-model exception, an unreachable URL) both
would have shown up as a `0.000`/`inf` row with a populated `error` field,
not this crash, which is specific to the empty-scenarios shape.
