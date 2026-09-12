#!/usr/bin/env python
"""
scripts/calibrate_confidence_threshold.py

Manual/offline tool: derive a real `[embedding].confidence_threshold` for a
configured remote embedding model, instead of guessing one.

Background (issue #46): `_SEMANTIC_CONFIDENCE_THRESHOLD` in server.py (now
`PDFConfig.confidence_threshold`'s built-in 0.5 default) is tuned to
BAAI/bge-small-en-v1.5's own cosine-similarity distribution. A model with a
different distribution (e5, nomic, Qwen3-Embedding, ...) needs its own
threshold -- PDFConfig.confidence_threshold returns None for any model this
codebase does not recognize as bge-small-compatible until one is set
explicitly. This script is how you get that number: it runs a configured
remote model over the existing hand-annotated ground truth
(benchmark_data/ground_truth.json), scores every (query, page) pair by
cosine similarity, and sweeps candidate thresholds to find the one that best
separates known-relevant pages from known-irrelevant ones.

This REQUIRES a live embedding backend reachable at --base-url: it makes
real HTTP requests (downloads each ground-truth PDF, then embeds every page
and every query against your endpoint). It is not run in CI, the same way
scripts/benchmark_embedding_models.py isn't -- see tests/test_calibrate_
confidence_threshold.py for the part of this module that IS unit-tested
without a live server: sweep_thresholds/best_threshold, the pure
scoring logic below.

Metric: F1 at each candidate threshold, chosen over "precision at a fixed
recall" because there is no natural recall target to fix here (unlike a
ranking benchmark with a known k) -- F1 picks the threshold that jointly
maximizes precision and recall without an arbitrary extra parameter, and
the sweep still reports precision/recall/support for the winner so you can
judge the trade-off directly. Ties broken by preferring the HIGHER
threshold: a false "confident" match (bad precision) actively misleads an
agent into trusting a bad excerpt, while a false "low_confidence" flag on a
genuinely good match (bad recall) only costs it a second look -- the safer
side to lean on when two candidates score identically.

Ground truth shape (benchmark_data/ground_truth.json):
    {"pdfs": {key: {"url", "title", "page_count",
                     "scenarios": {id: {"query", "relevant_pages": [int, ...]}}}}}
`relevant_pages` are 1-indexed. Every OTHER page in the same document is
treated as irrelevant for that query -- a simplifying assumption (a page
neither confirmed relevant nor annotated is assumed irrelevant), documented
here rather than silently baked in; it is the same assumption
scripts/benchmark_embedding_models.py's recall/RR metrics rest on.

Run:
    python scripts/calibrate_confidence_threshold.py \\
        --base-url http://localhost:8000/v1 \\
        --model nomic-embed-text \\
        --query-prefix "search_query: " \\
        --document-prefix "search_document: "

Prints the suggested confidence_threshold and the precision/recall/F1 it
achieves on the ground-truth set, plus the full sweep table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402

from pdf_mcp.remote_embedder import RemoteSpec  # noqa: E402
from pdf_mcp.remote_embedder import encode as remote_encode  # noqa: E402

# ── Pure scoring / sweep logic (unit-tested without a live server) ─────────


def _l2_normalize(vecs: Any) -> Any:
    """Row-wise L2 normalize so a dot product equals cosine similarity.

    remote_embedder.encode returns UNNORMALIZED vectors by design (servers
    differ in whether they already return unit vectors) -- embedder.py
    normalizes for the real search path, and this script must do the same
    to compute a comparable cosine.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def score_at_threshold(
    pairs: list[tuple[float, bool]], threshold: float
) -> dict[str, float]:
    """
    Precision/recall/F1 of "cosine >= threshold" as a predictor of
    `is_relevant`, over `pairs` of (cosine_similarity, is_relevant).

    "cosine >= threshold" is the PDF-mcp semantics inverted: `low_confidence`
    is `score < threshold`, so a match is NOT low_confidence (i.e. predicted
    relevant / trustworthy) exactly when `score >= threshold`.

    Returns a dict with precision, recall, f1 (0.0 when a denominator is 0,
    matching the conventional scikit-learn `zero_division=0` default: a
    threshold with no positive predictions has 0.0 precision, and one with
    no actual positives to find has 0.0 recall) plus tp/fp/fn/tn counts for
    inspection.
    """
    tp = fp = fn = tn = 0
    for cosine, is_relevant in pairs:
        predicted_relevant = cosine >= threshold
        if predicted_relevant and is_relevant:
            tp += 1
        elif predicted_relevant and not is_relevant:
            fp += 1
        elif not predicted_relevant and is_relevant:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def sweep_thresholds(
    pairs: list[tuple[float, bool]],
    thresholds: "list[float] | None" = None,
) -> list[dict[str, float]]:
    """score_at_threshold for every candidate in `thresholds`.

    Default sweep: -1.00 to 1.00 in steps of 0.01 (201 candidates) -- finer
    than a typical cosine's meaningful precision needs to be, but cheap
    (pure Python over an in-memory list; no re-embedding per candidate).
    """
    if thresholds is None:
        thresholds = [round(-1.0 + 0.01 * i, 2) for i in range(201)]
    return [score_at_threshold(pairs, t) for t in thresholds]


def best_threshold(
    pairs: list[tuple[float, bool]],
    thresholds: "list[float] | None" = None,
) -> dict[str, float]:
    """The sweep_thresholds entry with the highest F1.

    Ties broken by the HIGHER threshold -- see the module docstring for
    why that's the safer direction to round in. Raises ValueError on an
    empty `pairs` (nothing to calibrate against).
    """
    if not pairs:
        raise ValueError("no (cosine, is_relevant) pairs to calibrate against")
    results = sweep_thresholds(pairs, thresholds)
    return max(results, key=lambda r: (r["f1"], r["threshold"]))


# ── Live ground-truth collection (requires a reachable --base-url) ────────


def _load_ground_truth(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    with open(path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


def _page_texts(pdf_path: str) -> dict[int, str]:
    """1-indexed page number -> extracted text, skipping pages with no
    extractable text (nothing to embed -- an OCR-only page tells us
    nothing about this model's cosine distribution)."""
    import pymupdf

    from pdf_mcp.extractor import extract_text_from_page

    texts: dict[int, str] = {}
    doc = pymupdf.open(pdf_path)
    try:
        for i in range(doc.page_count):
            text = extract_text_from_page(doc[i]).strip()
            if text:
                texts[i + 1] = text
    finally:
        doc.close()
    return texts


def collect_pairs(
    ground_truth: dict[str, Any],
    spec: RemoteSpec,
    *,
    resolve_pdf: Any,
    print_progress: bool = True,
) -> list[tuple[float, bool]]:
    """
    Build (cosine_similarity, is_relevant) pairs for every (query, page)
    combination across every scenario in `ground_truth`.

    `resolve_pdf(url) -> local_path` is injected (rather than importing
    server._resolve_path directly) so this stays testable with a fake --
    the real CLI passes pdf_mcp.server._resolve_path, which downloads via
    this codebase's existing SSRF-hardened URLFetcher and local cache.
    """
    pairs: list[tuple[float, bool]] = []
    for pdf_key, pdf in ground_truth.get("pdfs", {}).items():
        scenarios = pdf.get("scenarios") or {}
        if not scenarios:
            continue
        local_path, err = resolve_pdf(pdf["url"])
        if err is not None:
            if print_progress:
                print(f"  ! skipping {pdf_key}: {err.get('error')}")
            continue
        page_texts = _page_texts(local_path)
        if not page_texts:
            if print_progress:
                print(f"  ! skipping {pdf_key}: no extractable text")
            continue
        pages = sorted(page_texts)
        doc_vecs = _l2_normalize(
            remote_encode(
                [page_texts[p] for p in pages], spec, prefix=spec.document_prefix
            )
        )
        page_vec = dict(zip(pages, doc_vecs))

        for sid, scenario in scenarios.items():
            query = scenario["query"]
            relevant = set(scenario["relevant_pages"])
            query_vec = _l2_normalize(
                remote_encode([query], spec, prefix=spec.query_prefix)
            )[0]
            for p in pages:
                cosine = float(page_vec[p] @ query_vec)
                pairs.append((cosine, p in relevant))
            if print_progress:
                print(f"  {pdf_key}/{sid}: {len(pages)} pages scored")
    return pairs


def _print_report(result: dict[str, float], n_pairs: int) -> None:
    print()
    print("=" * 60)
    print("  Suggested [embedding].confidence_threshold")
    print("=" * 60)
    print(f"  threshold  = {result['threshold']:.2f}")
    print(f"  precision  = {result['precision']:.3f}")
    print(f"  recall     = {result['recall']:.3f}")
    print(f"  f1         = {result['f1']:.3f}")
    print(
        f"  support    = {n_pairs} (query, page) pairs "
        f"(tp={result['tp']:.0f} fp={result['fp']:.0f} "
        f"fn={result['fn']:.0f} tn={result['tn']:.0f})"
    )
    print()
    print("Paste into config.toml:")
    print()
    print("  [embedding]")
    print(f"  confidence_threshold = {result['threshold']:.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--ground-truth",
        default=str(
            Path(__file__).parent.parent / "benchmark_data" / "ground_truth.json"
        ),
    )
    args = parser.parse_args()

    import os

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    spec = RemoteSpec(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        document_prefix=args.document_prefix,
        query_prefix=args.query_prefix,
        dimensions=args.dimensions,
    )

    ground_truth = _load_ground_truth(Path(args.ground_truth))

    # server._resolve_path downloads/caches via this codebase's own
    # SSRF-hardened URLFetcher and honors [paths]/[urls] config -- the same
    # helper scripts/benchmark_embedding_models.py uses for the same
    # ground-truth file.
    import pdf_mcp.server as server_module

    print(
        f"Calibrating confidence_threshold for model={args.model!r} "
        f"at {args.base_url!r} ..."
    )
    pairs = collect_pairs(ground_truth, spec, resolve_pdf=server_module._resolve_path)
    if not pairs:
        print("No (query, page) pairs collected -- nothing to calibrate.")
        sys.exit(1)

    result = best_threshold(pairs)
    _print_report(result, len(pairs))


if __name__ == "__main__":
    main()
