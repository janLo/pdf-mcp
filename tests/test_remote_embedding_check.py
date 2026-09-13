"""Tests for pdf_mcp.remote_embedding_check (the startup cosine-parity
safety check, issue #42).

No HTTP mocking here: `verify_remote_backend` takes an injectable
`encode_fn` (same signature as `remote_embedder.encode`), so these tests
never touch httpx or a live server -- they exercise the comparison logic
directly against the real, committed `bge_small_reference.json`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pdf_mcp.remote_embedding_check import (
    REFERENCE_PATH,
    load_reference,
    verify_remote_backend,
)
from pdf_mcp.remote_embedder import RemoteSpec


def _spec(**overrides) -> RemoteSpec:
    defaults = dict(base_url="http://localhost:8712/v1", model="bge-small-en-v1.5")
    defaults.update(overrides)
    return RemoteSpec(**defaults)


class TestLoadReference:
    def test_reference_file_exists_and_loads(self):
        sentences, vectors = load_reference()
        assert len(sentences) >= 5
        assert vectors.shape == (len(sentences), 384)

    def test_reference_vectors_are_unit_norm(self):
        _, vectors = load_reference()
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_reference_path_is_shipped_in_the_package(self):
        assert REFERENCE_PATH.exists()
        assert REFERENCE_PATH.name == "bge_small_reference.json"


class TestVerifyRemoteBackendMatching:
    def test_identical_vectors_pass(self):
        sentences, vectors = load_reference()

        def fake_encode(texts, spec):
            assert texts == sentences
            return vectors.copy()

        result = verify_remote_backend(_spec(), encode_fn=fake_encode)
        assert result.ok
        assert result.min_cosine == pytest.approx(1.0, abs=1e-5)
        assert result.mean_cosine == pytest.approx(1.0, abs=1e-5)

    def test_unnormalized_but_parallel_vectors_still_pass(self):
        """encode_fn returns UNNORMALIZED vectors (matches
        remote_embedder.encode's real contract) -- the check must
        normalize before comparing, not assume unit vectors."""
        sentences, vectors = load_reference()

        def fake_encode(texts, spec):
            return vectors * 7.3  # same direction, different magnitude

        result = verify_remote_backend(_spec(), encode_fn=fake_encode)
        assert result.ok
        assert result.min_cosine == pytest.approx(1.0, abs=1e-4)

    def test_small_noise_within_threshold_passes(self):
        sentences, vectors = load_reference()
        rng = np.random.default_rng(0)
        noisy = vectors + rng.normal(scale=0.001, size=vectors.shape).astype(np.float32)

        def fake_encode(texts, spec):
            return noisy

        result = verify_remote_backend(_spec(), encode_fn=fake_encode)
        assert result.ok
        assert result.min_cosine >= 0.99


class TestVerifyRemoteBackendFallback:
    def test_sufficiently_different_vectors_fail(self):
        sentences, vectors = load_reference()
        rng = np.random.default_rng(1)
        # Replace one row entirely with unrelated random noise -- simulates
        # a wrong-model/wrong-pooling endpoint on at least one sentence.
        wrong = vectors.copy()
        wrong[0] = rng.normal(size=vectors.shape[1]).astype(np.float32)

        def fake_encode(texts, spec):
            return wrong

        result = verify_remote_backend(_spec(), encode_fn=fake_encode)
        assert not result.ok
        assert result.min_cosine < 0.99
        assert "cosine similarity" in result.reason

    def test_all_random_vectors_fail(self):
        sentences, vectors = load_reference()
        rng = np.random.default_rng(2)
        random_vecs = rng.normal(size=vectors.shape).astype(np.float32)

        def fake_encode(texts, spec):
            return random_vecs

        result = verify_remote_backend(_spec(), encode_fn=fake_encode)
        assert not result.ok
        assert result.mean_cosine is not None
        assert result.mean_cosine < 0.5

    def test_wrong_dimension_fails_without_raising(self):
        sentences, vectors = load_reference()

        def fake_encode(texts, spec):
            return vectors[:, :10]  # wrong dimensionality entirely

        result = verify_remote_backend(_spec(), encode_fn=fake_encode)
        assert not result.ok
        assert "shape" in result.reason

    def test_network_failure_fails_without_raising(self):
        def fake_encode(texts, spec):
            raise ConnectionError("connection refused")

        result = verify_remote_backend(_spec(), encode_fn=fake_encode)
        assert not result.ok
        assert "connection refused" in result.reason
        assert result.mean_cosine is None

    def test_corrupt_reference_file_fails_without_raising(self, tmp_path):
        bad_path = tmp_path / "bad_reference.json"
        bad_path.write_text("not json", encoding="utf-8")

        def fake_encode(texts, spec):
            raise AssertionError("must not be reached: load_reference failed first")

        result = verify_remote_backend(
            _spec(), encode_fn=fake_encode, reference_path=bad_path
        )
        assert not result.ok
        assert "could not embed reference sentences" in result.reason

    def test_empty_reference_file_fails_without_raising(self, tmp_path):
        empty_path = tmp_path / "empty_reference.json"
        empty_path.write_text(
            json.dumps({"sentences": [], "vectors": []}), encoding="utf-8"
        )

        def fake_encode(texts, spec):
            raise AssertionError("must not be reached: empty reference rejected first")

        result = verify_remote_backend(
            _spec(), encode_fn=fake_encode, reference_path=empty_path
        )
        assert not result.ok

    def test_check_uses_a_short_single_batch_spec_not_the_caller_s(self):
        """The startup check must not inherit the caller's bulk timeout/
        batch_size/max_concurrency (see module docstring) -- it should call
        encode_fn with its own short-timeout, single-batch, no-concurrency
        spec regardless of what was passed in."""
        seen_specs = []

        def fake_encode(texts, spec):
            seen_specs.append(spec)
            return load_reference()[1]

        caller_spec = _spec(timeout=60.0, batch_size=32, max_concurrency=4)
        verify_remote_backend(caller_spec, encode_fn=fake_encode)

        assert len(seen_specs) == 1
        used = seen_specs[0]
        assert used.timeout < caller_spec.timeout
        assert used.max_concurrency == 1
        assert used.max_attempts == 1  # PR #47 review item 3: fail fast
        assert used.base_url == caller_spec.base_url  # everything else preserved
        assert used.model == caller_spec.model

    def test_custom_threshold_is_honored(self):
        sentences, vectors = load_reference()
        rng = np.random.default_rng(3)
        noisy = vectors + rng.normal(scale=0.05, size=vectors.shape).astype(np.float32)

        def fake_encode(texts, spec):
            return noisy

        loose = verify_remote_backend(_spec(), threshold=0.5, encode_fn=fake_encode)
        strict = verify_remote_backend(_spec(), threshold=0.9999, encode_fn=fake_encode)
        assert loose.ok
        assert not strict.ok
