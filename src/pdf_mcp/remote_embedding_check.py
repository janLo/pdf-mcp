"""
Startup safety check for the remote ("openai:"-identity) embedding backend.

pdf-mcp cannot see which model actually sits behind a configured
``[embedding].base_url`` -- an OpenAI-compatible ``/v1/embeddings`` endpoint
can silently serve the wrong model, a different quantization, a different
pooling strategy, or nothing at all (wrong port, stale reverse proxy). Since
the whole point of the remote backend is to stay in the same vector space as
local fastembed ``BAAI/bge-small-en-v1.5`` (so cached vectors, `low_confidence`
thresholds, and hybrid RRF fusion tuned against that space stay valid), a
silent mismatch would corrupt search quality without any visible error.

This module embeds a small fixed set of reference sentences
(`REFERENCE_PATH`, precomputed once via local fastembed --
see `scripts/gen_bge_small_reference.py`) through the *remote* endpoint and
compares each resulting vector to its stored fastembed reference by cosine
similarity. See issue #42: "It would
embed a few fixed sentences, compare them against stored fastembed bge-small
reference vectors, and fall back to CPU with a warning if cosine drops below
~0.99."

Aggregate rule: the MINIMUM per-sentence cosine must clear the threshold, not
just the mean -- a single sentence with a low cosine (e.g. one long input
truncated differently by a mismatched context window) is exactly the kind of
localized failure a mean could hide. Threshold default 0.99, matching the
issue's own number.

Called once at server startup (server.py) right before
``embedder.configure_remote``. Never raises for a reachability/format
failure -- a startup check that could crash the server on a flaky network
would be worse than the problem it guards against; it reports ``ok=False``
with a human-readable reason instead, and the caller decides to fall back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .remote_embedder import RemoteSpec

REFERENCE_PATH = Path(__file__).with_name("bge_small_reference.json")

# jztan's number (issue #42): "fall back to CPU with a warning if cosine
# drops below ~0.99".
DEFAULT_THRESHOLD = 0.99

# The type of remote_embedder.encode: (texts, spec) -> ndarray (N, D),
# UNNORMALIZED. Exposed here so tests can inject a fake without touching
# httpx.
RemoteEncodeFn = Callable[["list[str]", RemoteSpec], Any]


@dataclass(frozen=True)
class SafetyCheckResult:
    """Outcome of one startup safety check run.

    ``ok`` is the single bit the caller (server.py) acts on. Everything
    else is diagnostic detail for the log message / tests.
    """

    ok: bool
    reason: str
    mean_cosine: "float | None" = None
    min_cosine: "float | None" = None
    per_sentence_cosine: "tuple[float, ...] | None" = None


def load_reference(path: Path = REFERENCE_PATH) -> tuple[list[str], Any]:
    """Load the stored (sentences, vectors) reference pair.

    ``vectors`` is an (N, D) float32 ndarray, L2-normalized (same convention
    as embedder.encode's output) -- so a plain dot product against a
    normalized remote vector equals cosine similarity.
    """
    import numpy as np

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences: list[str] = data["sentences"]
    vectors = np.array(data["vectors"], dtype=np.float32)
    return sentences, vectors


def verify_remote_backend(
    spec: RemoteSpec,
    threshold: float = DEFAULT_THRESHOLD,
    encode_fn: "RemoteEncodeFn | None" = None,
    reference_path: Path = REFERENCE_PATH,
) -> SafetyCheckResult:
    """Embed the stored reference sentences via `spec` and compare to the
    stored fastembed vectors.

    `encode_fn` defaults to `remote_embedder.encode`; tests inject a fake
    that returns pre-baked vectors so no HTTP call is made. Any exception
    from `encode_fn` (network error, bad JSON, dimension mismatch, ...) is
    caught and reported as `ok=False` with the exception text as `reason`
    -- see the module docstring for why this never raises.
    """
    import numpy as np

    if encode_fn is None:
        from . import remote_embedder

        encode_fn = remote_embedder.encode

    sentences, reference_vecs = load_reference(reference_path)

    try:
        remote_vecs = encode_fn(sentences, spec)
    except Exception as exc:  # noqa: BLE001 - reported, never propagated
        return SafetyCheckResult(
            ok=False,
            reason=f"could not embed reference sentences via the remote "
            f"endpoint: {exc!r}",
        )

    remote_vecs = np.asarray(remote_vecs, dtype=np.float32)
    if remote_vecs.shape != reference_vecs.shape:
        return SafetyCheckResult(
            ok=False,
            reason=(
                f"remote endpoint returned vectors of shape "
                f"{remote_vecs.shape}, expected {reference_vecs.shape} "
                f"({len(sentences)} sentences x {reference_vecs.shape[1]} "
                "dims) -- likely a different model or dimensionality "
                "behind the endpoint"
            ),
        )

    norms = np.linalg.norm(remote_vecs, axis=1, keepdims=True)
    remote_unit = remote_vecs / np.clip(norms, 1e-12, None)
    # reference_vecs is already unit-normalized (see load_reference).
    cosines = np.sum(remote_unit * reference_vecs, axis=1)
    mean_cos = float(np.mean(cosines))
    min_cos = float(np.min(cosines))

    if min_cos < threshold:
        worst_idx = int(np.argmin(cosines))
        return SafetyCheckResult(
            ok=False,
            reason=(
                f"cosine similarity to stored fastembed bge-small "
                f"reference vectors dropped below {threshold} (min="
                f"{min_cos:.4f}, mean={mean_cos:.4f}, worst sentence: "
                f"{sentences[worst_idx]!r}) -- the remote endpoint may be "
                "serving a different model, quantization, or pooling "
                "strategy than fastembed's BAAI/bge-small-en-v1.5"
            ),
            mean_cosine=mean_cos,
            min_cosine=min_cos,
            per_sentence_cosine=tuple(float(c) for c in cosines),
        )

    return SafetyCheckResult(
        ok=True,
        reason=f"cosine parity OK (min={min_cos:.4f}, mean={mean_cos:.4f})",
        mean_cosine=mean_cos,
        min_cosine=min_cos,
        per_sentence_cosine=tuple(float(c) for c in cosines),
    )
