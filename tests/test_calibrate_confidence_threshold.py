# tests/test_calibrate_confidence_threshold.py
"""
Unit tests for scripts/calibrate_confidence_threshold.py's pure threshold-
sweep/scoring logic (score_at_threshold, sweep_thresholds, best_threshold).

These run against small synthetic (cosine, is_relevant) sets -- no live
embedding backend, no network, no PDF download. The script's live
ground-truth collection path (collect_pairs, main) needs a reachable
--base-url and is exercised manually, not here -- see the module docstring.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import calibrate_confidence_threshold as cct  # noqa: E402


class TestScoreAtThreshold:
    def test_perfectly_separated_pairs(self):
        """Relevant pages all score high, irrelevant all score low: a
        threshold between the two clusters gets perfect precision/recall."""
        pairs = [
            (0.9, True),
            (0.8, True),
            (0.85, True),
            (0.1, False),
            (0.2, False),
            (0.05, False),
        ]
        result = cct.score_at_threshold(pairs, 0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (3, 0, 0, 3)

    def test_threshold_too_high_hurts_recall_not_precision(self):
        pairs = [(0.9, True), (0.6, True), (0.1, False)]
        # threshold above the second relevant page's score misses it
        result = cct.score_at_threshold(pairs, 0.7)
        assert result["tp"] == 1
        assert result["fn"] == 1
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5

    def test_threshold_too_low_hurts_precision_not_recall(self):
        pairs = [(0.9, True), (0.6, False), (0.55, False)]
        result = cct.score_at_threshold(pairs, 0.5)
        assert result["tp"] == 1
        assert result["fp"] == 2
        assert result["recall"] == 1.0
        assert result["precision"] == pytest.approx(1 / 3)

    def test_no_positive_predictions_gives_zero_precision(self):
        """threshold above every score -> nothing predicted relevant ->
        precision is 0.0 (zero_division=0 convention), not NaN/error."""
        pairs = [(0.1, True), (0.2, False)]
        result = cct.score_at_threshold(pairs, 5.0)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["precision"] == 0.0
        assert result["f1"] == 0.0

    def test_no_actual_positives_gives_zero_recall(self):
        """No relevant pages in this slice at all -> recall 0.0, not
        division by zero."""
        pairs = [(0.9, False), (0.1, False)]
        result = cct.score_at_threshold(pairs, 0.5)
        assert result["recall"] == 0.0


class TestSweepThresholds:
    def test_default_sweep_spans_cosine_range(self):
        pairs = [(0.5, True), (-0.5, False)]
        results = cct.sweep_thresholds(pairs)
        thresholds = [r["threshold"] for r in results]
        assert min(thresholds) == -1.0
        assert max(thresholds) == 1.0
        assert len(results) == 201

    def test_custom_thresholds_used_verbatim(self):
        pairs = [(0.5, True), (-0.5, False)]
        results = cct.sweep_thresholds(pairs, thresholds=[0.0, 0.3, 0.9])
        assert [r["threshold"] for r in results] == [0.0, 0.3, 0.9]


class TestBestThreshold:
    def test_picks_highest_f1(self):
        pairs = [
            (0.9, True),
            (0.85, True),
            (0.1, False),
            (0.15, False),
        ]
        result = cct.best_threshold(pairs, thresholds=[0.0, 0.5, 1.0])
        assert result["threshold"] == 0.5
        assert result["f1"] == 1.0

    def test_ties_broken_by_higher_threshold(self):
        """Two thresholds land in the same gap and score identically ->
        the higher one wins (see module docstring: a false 'confident'
        match is worse than a false 'low_confidence' one)."""
        pairs = [(0.9, True), (0.1, False)]
        result = cct.best_threshold(pairs, thresholds=[0.3, 0.5, 0.7])
        assert result["f1"] == 1.0
        assert result["threshold"] == 0.7

    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError, match="no .* pairs"):
            cct.best_threshold([])

    def test_realistic_mixed_distribution(self):
        """A less clean synthetic set (some overlap between the two
        clusters) still yields a sane, in-range threshold rather than an
        edge value."""
        pairs = (
            [(0.75 + 0.01 * i, True) for i in range(10)]
            + [(0.55 + 0.01 * i, True) for i in range(3)]  # a few harder positives
            + [(0.20 + 0.01 * i, False) for i in range(10)]
            + [(0.45 + 0.01 * i, False) for i in range(2)]  # a couple hard negatives
        )
        result = cct.best_threshold(pairs)
        assert -1.0 < result["threshold"] < 1.0
        assert result["f1"] > 0.8


class TestCollectPairs:
    def test_uses_injected_resolver_and_skips_failures(self, tmp_path, monkeypatch):
        """collect_pairs never touches the network itself -- resolve_pdf is
        fully injected, so this is exercisable without a live server."""
        import numpy as np

        # A single real (but tiny, synthetic) one-page PDF via PyMuPDF, so
        # extract_text_from_page has something to find.
        import pymupdf

        pdf_path = tmp_path / "doc.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "hello world")
        doc.save(str(pdf_path))
        doc.close()

        ground_truth = {
            "pdfs": {
                "ok": {
                    "url": "https://example.invalid/ok.pdf",
                    "scenarios": {"1a": {"query": "hello", "relevant_pages": [1]}},
                },
                "unresolvable": {
                    "url": "https://example.invalid/missing.pdf",
                    "scenarios": {"1a": {"query": "x", "relevant_pages": [1]}},
                },
                "no_scenarios": {
                    "url": "https://example.invalid/other.pdf",
                    "scenarios": {},
                },
            }
        }

        def fake_resolve(url):
            if "missing" in url:
                return None, {"error": "not found"}
            return str(pdf_path), None

        seen_prefixes = []

        def fake_encode(texts, spec, *, prefix=""):
            seen_prefixes.append(prefix)
            # 2D and content-dependent (length, vowel count) rather than a
            # single dimension -- a 1-D vector L2-normalizes to +/-1.0 and
            # makes every cosine trivially 1.0 regardless of whether the
            # comparison logic is even correct.
            return np.array(
                [
                    [float(len(t)), float(sum(t.lower().count(v) for v in "aeiou"))]
                    for t in texts
                ],
                dtype=np.float32,
            )

        monkeypatch.setattr(cct, "remote_encode", fake_encode)

        spec = cct.RemoteSpec(
            base_url="http://localhost:8000/v1",
            model="fake-model",
            document_prefix="DOC: ",
            query_prefix="Q: ",
        )
        pairs = cct.collect_pairs(
            ground_truth, spec, resolve_pdf=fake_resolve, print_progress=False
        )
        # Only "ok" contributes (1 page x 1 scenario); the unresolvable and
        # scenario-less entries are skipped without raising.
        assert len(pairs) == 1
        cosine, is_relevant = pairs[0]
        assert is_relevant is True
        # A real, computed cosine between "hello world"'s and "hello"'s
        # (length, vowel-count) vectors -- not the degenerate +/-1.0 a
        # 1-D vector would always produce regardless of correctness.
        page_vec = np.array([11.0, 3.0])  # "hello world": len 11, 3 vowels
        query_vec = np.array([5.0, 2.0])  # "hello": len 5, 2 vowels
        expected = float(
            page_vec
            @ query_vec
            / (np.linalg.norm(page_vec) * np.linalg.norm(query_vec))
        )
        assert cosine == pytest.approx(expected, abs=1e-5)

        # document_prefix and query_prefix must each reach the right call
        # (a swap here would silently corrupt every calibration run).
        assert "DOC: " in seen_prefixes
        assert "Q: " in seen_prefixes
