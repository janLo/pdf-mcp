#!/usr/bin/env python
"""
scripts/benchmark_remote_embedder.py

Measures the SHIPPED `pdf_mcp.remote_embedder.encode()` against a real
OpenAI-compatible embeddings endpoint (lemonade, ollama, llama-server, ...).
Requires a server already running and reachable -- this script makes no
attempt to start one.

Two things it measures, both against synthetic text (real PDF page content
is not needed to measure network+backend throughput):

  1. Throughput across `--concurrency` levels (default 1,2,4,8), compared
     against the local fastembed CPU baseline (`BAAI/bge-small-en-v1.5`) for
     scale, NOT as an apples-to-apples backend comparison unless
     `--model-params-note` in the output is heeded -- see
     benchmark_data/remote_embedder_results.md for why model size confounds
     that comparison unless the remote model is also tiny.
  2. A correctness/sanity check: vectors are unit-norm (L2, applied by
     embedder.py, not this script), same shape across a batch, and a query
     against a 3-sentence toy corpus ranks the obviously-relevant sentence
     first.

Run:
    uv run python scripts/benchmark_remote_embedder.py \
        --base-url http://localhost:13305/v1 --model Qwen3-Embedding-0.6B-GGUF

Always exits 0 (informational; no CI gate -- there is no server to hit in CI).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pdf_mcp import embedder  # noqa: E402
from pdf_mcp.remote_embedder import RemoteSpec, encode  # noqa: E402


def _synthetic_passages(n: int) -> list[str]:
    return [
        f"This is benchmark passage number {i}. It discusses cloud "
        "computing, PDF extraction, and government policy in moderate "
        "detail to approximate a real page chunk."
        for i in range(n)
    ]


def run_throughput(spec_base: RemoteSpec, texts: list[str], levels: list[int]) -> None:
    print(f"\n=== Throughput: {spec_base.model} @ {spec_base.base_url} ===")
    for conc in levels:
        spec = RemoteSpec(
            base_url=spec_base.base_url,
            model=spec_base.model,
            api_key=spec_base.api_key,
            timeout=spec_base.timeout,
            batch_size=spec_base.batch_size,
            max_concurrency=conc,
        )
        t0 = time.monotonic()
        arr = encode(texts, spec)
        dt = time.monotonic() - t0
        print(
            f"  concurrency={conc}: {dt:.2f}s ({len(texts) / dt:.1f} texts/s), "
            f"dim={arr.shape[1]}"
        )

    print("\n  -- fastembed CPU baseline (BAAI/bge-small-en-v1.5) --")
    t0 = time.monotonic()
    arr = embedder.encode(texts, "BAAI/bge-small-en-v1.5")
    dt = time.monotonic() - t0
    print(f"  {dt:.2f}s ({len(texts) / dt:.1f} texts/s), dim={arr.shape[1]}")
    print(
        "  NOTE: not apples-to-apples unless the remote model is also ~33M "
        "params -- see benchmark_data/remote_embedder_results.md."
    )


def run_correctness(spec: RemoteSpec) -> bool:
    print(f"\n=== Correctness: {spec.model} @ {spec.base_url} ===")
    import numpy as np

    docs = [
        "The cat sat on the mat.",
        "Quantum computing uses qubits.",
        "PDF files store text and images.",
    ]
    doc_vecs = encode(docs, spec)
    ok = True
    if doc_vecs.shape[0] != len(docs):
        print(f"  FAIL: expected {len(docs)} rows, got {doc_vecs.shape[0]}")
        ok = False

    query_vecs = encode(["What stores images?"], spec, prefix=spec.query_prefix)
    q = query_vecs[0]
    # This script's own vectors are not normalized (remote_embedder never
    # normalizes -- embedder.py owns that); normalize here purely to score.
    doc_norm = doc_vecs / np.clip(
        np.linalg.norm(doc_vecs, axis=1, keepdims=True), 1e-12, None
    )
    q_norm = q / max(float(np.linalg.norm(q)), 1e-12)
    sims = doc_norm @ q_norm
    best = docs[int(np.argmax(sims))]
    print(f"  similarities: {sims}")
    print(f"  best match: {best!r}")
    if best != docs[2]:
        print("  FAIL: expected the PDF/images sentence to rank first")
        ok = False
    else:
        print("  OK: correct sentence ranked first")
    return ok


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:13305/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-texts", type=int, default=64)
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args(argv)

    spec = RemoteSpec(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        batch_size=args.batch_size,
        query_prefix=args.query_prefix,
    )

    if not args.skip_correctness:
        run_correctness(spec)
    if not args.skip_throughput:
        run_throughput(spec, _synthetic_passages(args.n_texts), args.concurrency)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
