"""
HTTP client for an OpenAI-compatible ``POST /v1/embeddings`` endpoint.

Covers ollama, lemonade, bare ``llama-server``, vLLM, and hosted providers
(OpenRouter, OpenAI) with one implementation, since they all speak the same
request/response schema. This module is intentionally the only place that
touches the network for embeddings; ``embedder.py`` owns dispatch and
normalization, ``config.py`` owns parsing ``[embedding]`` into a `RemoteSpec`.

This branch (see issue #46) extends the narrower first version (issue #42,
`feature/embed-remote-narrow`) with support for an arbitrary remote model:
`RemoteSpec.document_prefix`/`query_prefix` (asymmetric prefix protocols
like nomic's "search_document:"/"search_query:") and `dimensions` (output
width validation). `model` is no longer purely a cache-naming label --
it now genuinely selects what gets encoded and how, which is exactly the
surface the upstream maintainer originally asked to keep out of the first
version: `_SEMANTIC_CONFIDENCE_THRESHOLD` and the hybrid RRF fusion in
server.py are tuned to bge-small-en-v1.5's own cosine-similarity
distribution, and nothing in this module (or anywhere else in this
codebase) validates that a configured model's distribution is compatible
with that tuning. See the TODO on `_SEMANTIC_CONFIDENCE_THRESHOLD` in
server.py -- recalibrating (or per-model-profiling) that threshold is
unresolved, tracked work, not something this branch attempts to fix.

Unlike the local fastembed path (one process, one encode call), a remote
endpoint is I/O-bound: batches are issued concurrently from a bounded thread
pool (``httpx.Client`` is thread-safe; `_get_model` in embedder.py is
explicitly documented as NOT thread-safe, which is why this stays a separate
client rather than a shared module global).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np

# Default attempts per batch: 1 initial + up to 2 retries. A prewarm run is
# thousands of requests against a server that may be a personal machine
# (rate limits, a cold model load, a GPU busy with something else) -- worth
# more resilience than url_fetcher.py's single bare retry (that one guards
# against one transient download corruption, not a sustained multi-minute
# run). `RemoteSpec.max_attempts` lets a caller override this -- see its
# docstring and remote_embedding_check.py, which wants exactly one attempt
# so an unreachable endpoint fails fast at startup instead of inheriting
# this budget.
MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5  # seconds, doubled each attempt, plus jitter
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RemoteEmbeddingError(RuntimeError):
    """The remote endpoint could not be used to produce embeddings."""


@dataclass(frozen=True)
class RemoteSpec:
    """Everything needed to call one OpenAI-compatible embeddings endpoint.

    ``api_key`` is the resolved secret value (already read from the env var
    named by ``[embedding].api_key_env``); this dataclass is never logged or
    included in an exception message in full -- see `_redact_base_url` and
    the header-only use in `_headers`.

    ``model`` names which model the remote server should load and is
    folded into the cache identity (see `identity_for`); embedder.py does
    not itself branch on its value, but `document_prefix`/`query_prefix`/
    `dimensions` below make the choice of model behaviorally significant
    -- see the module docstring for why that reopens issue #42's concern
    and the TODO on `_SEMANTIC_CONFIDENCE_THRESHOLD` in server.py.

    ``document_prefix``/``query_prefix`` are prepended to document and
    query text respectively before encoding (see `encode`'s `prefix`
    argument) -- required by some models' training protocol (nomic,
    Qwen3-Embedding), a no-op ("") for models like bge-small that use none.

    ``dimensions`` is optional; when set, `encode` validates the response
    width against it rather than silently accepting whatever width the
    server returns.

    ``max_attempts`` overrides `MAX_ATTEMPTS` for this spec -- used by
    remote_embedding_check.py to make exactly one attempt at startup rather
    than inheriting the bulk-embedding retry budget.
    """

    base_url: str
    model: str
    api_key: "str | None" = None
    timeout: float = 60.0
    batch_size: int = 32
    max_concurrency: int = 4
    max_attempts: int = MAX_ATTEMPTS
    dimensions: "int | None" = None
    document_prefix: str = ""
    query_prefix: str = ""


def identity_for(spec: RemoteSpec) -> str:
    """The cache-identity string for `spec`: 'openai:<host>[:<port>]/<model>
    [@<prefix-hash>]'.

    Pure function of `spec` (no I/O), so both PDFConfig.embedding_model and
    any future `--model` override can share it rather than reimplementing
    the same string.

    Host/port and the prefix hash are part of the identity deliberately --
    see PDFConfig.embedding_model's docstring for the full rationale
    (different endpoint or different prefixes = a different vector space,
    must not share a cache row). The API key is intentionally excluded,
    and so is any userinfo (user:pass@) a base_url might embed -- this
    identity string is stored in the cache DB, so it must be as
    credential-free as an error message (see `_redact_base_url`).
    """
    import hashlib

    parts = urlsplit(spec.base_url)
    host = parts.hostname or spec.base_url
    if parts.port:
        host = f"{host}:{parts.port}"
    identity = f"openai:{host}/{spec.model}"
    if spec.document_prefix or spec.query_prefix:
        digest = hashlib.sha256(
            f"{spec.document_prefix}\x00{spec.query_prefix}".encode("utf-8")
        ).hexdigest()[:8]
        identity += f"@{digest}"
    return identity


def _redact_base_url(base_url: str) -> str:
    """Strip userinfo (user:pass@) before a URL ever reaches a log or error
    message -- a base_url is user config and could embed credentials."""
    parts = urlsplit(base_url)
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _headers(spec: RemoteSpec) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if spec.api_key:
        headers["Authorization"] = f"Bearer {spec.api_key}"
    return headers


def _endpoint_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/embeddings"


def _post_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    """POST with retry on connect errors, timeouts, 429 and 5xx.

    `max_attempts` defaults to the bulk-embedding budget (`MAX_ATTEMPTS`)
    but is overridden per-spec (`RemoteSpec.max_attempts`) -- e.g.
    remote_embedding_check.py wants exactly 1, so a startup check fails
    fast on an unreachable endpoint instead of blocking for minutes.

    Never includes the Authorization header's value in a raised message --
    only the redacted URL and the response body/status, so a leaked
    exception (logs, an MCP error payload) cannot carry the API key.
    """
    last_exc: "Exception | None" = None
    for attempt in range(max_attempts):
        try:
            resp = client.post(url, json=payload, headers=headers)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_exc = exc
        else:
            if resp.status_code == 200:
                result: dict[str, Any] = resp.json()
                return result
            if resp.status_code not in _RETRYABLE_STATUS:
                raise RemoteEmbeddingError(
                    f"embedding request to {_redact_base_url(url)} failed: "
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )
            last_exc = RemoteEmbeddingError(
                f"HTTP {resp.status_code}: {resp.text[:500]}"
            )
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None and attempt < max_attempts - 1:
                try:
                    time.sleep(max(0.0, float(retry_after)))
                    continue
                except ValueError:
                    pass  # not a numeric seconds value; fall through to backoff
        if attempt < max_attempts - 1:
            delay = _RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
    raise RemoteEmbeddingError(
        f"embedding request to {_redact_base_url(url)} failed after "
        f"{max_attempts} attempts: {last_exc!r}"
    )


def _embed_batch(
    client: httpx.Client, spec: RemoteSpec, texts: list[str], prefix: str
) -> list[list[float]]:
    """One request for one batch. Returns rows in the CALLER's order.

    The OpenAI schema's `data[]` carries an `index` per item and does not
    guarantee response order matches request order -- reorder by it rather
    than trusting array position, and verify the count matches the batch
    (a server silently dropping an oversized/empty input would otherwise
    desync corpus.py's per-page unit slicing, which relies on positional
    alignment).
    """
    prefixed = [prefix + t for t in texts] if prefix else texts
    payload: dict[str, Any] = {
        "model": spec.model,
        "input": prefixed,
        # Several servers (notably some OpenAI-compatible proxies) default
        # to base64; this codebase's whole read path is
        # np.frombuffer(blob, dtype=np.float32) with no header, so a
        # silently base64-or-float64 response would be misread as garbage
        # rather than fail loudly. Ask for floats explicitly.
        "encoding_format": "float",
    }
    if spec.dimensions is not None:
        payload["dimensions"] = spec.dimensions
    body = _post_with_retry(
        client,
        _endpoint_url(spec.base_url),
        payload,
        _headers(spec),
        max_attempts=spec.max_attempts,
    )
    data = body.get("data")
    if not isinstance(data, list):
        raise RemoteEmbeddingError(
            f"embedding response from {_redact_base_url(spec.base_url)} "
            f"has no usable 'data' list: {str(body)[:300]}"
        )
    if len(data) != len(texts):
        raise RemoteEmbeddingError(
            f"embedding response returned {len(data)} vectors for "
            f"{len(texts)} inputs (model={spec.model!r})"
        )
    try:
        ordered = sorted(data, key=lambda item: item["index"])
        rows: list[list[float]] = [item["embedding"] for item in ordered]
    except (KeyError, TypeError) as exc:
        raise RemoteEmbeddingError(
            f"embedding response from {_redact_base_url(spec.base_url)} "
            f"is missing 'index'/'embedding' fields: {exc!r}"
        ) from exc
    return rows


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def encode(texts: list[str], spec: RemoteSpec, *, prefix: str = "") -> Any:
    """
    Encode `texts` via the OpenAI-compatible endpoint in `spec`.

    Returns an ndarray of shape (N, D), dtype float32, UNNORMALIZED --
    embedder.py owns L2 normalization for both the local and remote paths
    (servers differ in whether they return unit vectors, same reasoning as
    the fastembed path).

    `prefix` is the caller's document_prefix or query_prefix (embedder.py
    picks which); prepended to every text before the request, never sent
    empty-string-only (`prefixed = texts` unchanged when prefix == "").

    Model choice here is real (see the module docstring): the caller is
    responsible for having accounted for whatever cosine distribution the
    configured model produces before trusting `low_confidence`/RRF scores
    computed from these vectors -- this function does not and cannot check
    that itself.
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    batches = _chunks(texts, max(1, spec.batch_size))
    results: "list[list[list[float]] | None]" = [None] * len(batches)

    if len(batches) == 1 or spec.max_concurrency <= 1:
        with httpx.Client(timeout=spec.timeout) as client:
            for i, batch in enumerate(batches):
                results[i] = _embed_batch(client, spec, batch, prefix)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with httpx.Client(timeout=spec.timeout) as client:
            with ThreadPoolExecutor(
                max_workers=min(spec.max_concurrency, len(batches))
            ) as pool:
                futures = {
                    pool.submit(_embed_batch, client, spec, batch, prefix): i
                    for i, batch in enumerate(batches)
                }
                for fut in as_completed(futures):
                    # Order preservation is contract-critical here too:
                    # results[i] keyed by submission index, not completion
                    # order, then flattened below in that same order.
                    results[futures[fut]] = fut.result()

    flat_rows: list[list[float]] = []
    for batch_rows in results:
        assert batch_rows is not None  # every future/branch above fills its slot
        flat_rows.extend(batch_rows)

    try:
        arr = np.array(flat_rows, dtype=np.float32)
    except ValueError as exc:
        # numpy>=1.24 raises rather than silently building an object array
        # when row lengths differ -- e.g. one server response mixed
        # dimensions mid-batch. Surface it as our own error type instead of
        # a bare numpy ValueError so it matches every other failure mode
        # here.
        raise RemoteEmbeddingError(
            f"embedding response from {_redact_base_url(spec.base_url)} "
            f"returned vectors of inconsistent length: {exc}"
        ) from exc
    if arr.ndim != 2:
        raise RemoteEmbeddingError(
            f"embedding response from {_redact_base_url(spec.base_url)} "
            f"produced ragged/empty vectors (shape={arr.shape})"
        )
    if spec.dimensions is not None and arr.shape[1] != spec.dimensions:
        raise RemoteEmbeddingError(
            f"embedding response dimension {arr.shape[1]} does not match "
            f"configured dimensions={spec.dimensions} for model "
            f"{spec.model!r} at {_redact_base_url(spec.base_url)}"
        )
    return arr
