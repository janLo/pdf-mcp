# tests/test_gen_german_ground_truth.py
"""Unit tests for scripts/gen_german_ground_truth.py.

All synthetic -- no PDF download, no real BGB.pdf. Mirrors the style of
tests/test_benchmark_cjk_keyword.py: pure functions over hand-built toc
lists and page-text dicts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import gen_german_ground_truth as ggt  # noqa: E402


class TestPageBoilerplate:
    def test_strips_multiline_service_header(self):
        raw = (
            "Ein Service des Bundesministerium der Justiz und für "
            "Verbraucherschutz\nsowie des Bundesamts für Justiz "
            "‒ www.gesetze-im-internet.de\n(1) Der Vertrag kommt "
            "zustande."
        )
        cleaned = ggt._PAGE_BOILERPLATE_RE.sub(" ", raw)
        assert "gesetze-im-internet" not in cleaned
        assert "Der Vertrag kommt zustande." in cleaned

    def test_strips_page_footer(self):
        raw = "Text hier.\n- Seite 205 von 490 -"
        cleaned = ggt._PAGE_BOILERPLATE_RE.sub(" ", raw)
        assert "Seite" not in cleaned


class TestParseAnchors:
    def test_parses_simple_norm(self):
        toc = [[1, "§ 1\xa0Beginn der Rechtsfähigkeit", 26]]
        anchors = ggt.parse_anchors(toc)
        assert anchors == {"1": (26, "Beginn der Rechtsfähigkeit")}

    def test_parses_letter_suffixed_norm(self):
        toc = [[1, "§ 611a\xa0Arbeitsvertrag", 200]]
        anchors = ggt.parse_anchors(toc)
        assert anchors == {"611a": (200, "Arbeitsvertrag")}

    def test_drops_weggefallen(self):
        toc = [
            [1, "§ 2\xa0Eintritt der Volljährigkeit", 26],
            [1, "§ 88\xa0(weggefallen)", 30],
        ]
        skips = ggt.SkipLog()
        anchors = ggt.parse_anchors(toc, skips)
        assert "88" not in anchors
        assert skips.counts["weggefallen"] == 1

    def test_drops_range_entries(self):
        toc = [[1, "§§ 3 bis 6\xa0(weggefallen)", 26]]
        skips = ggt.SkipLog()
        anchors = ggt.parse_anchors(toc, skips)
        assert anchors == {}
        assert skips.counts["range_entry"] == 1

    def test_ignores_structural_entries(self):
        toc = [
            [1, "Buch 1", 25],
            [2, "Abschnitt 1", 25],
            [1, "§ 1\xa0Test", 26],
        ]
        anchors = ggt.parse_anchors(toc)
        assert list(anchors.keys()) == ["1"]

    def test_last_entry_wins_on_duplicate_number(self):
        toc = [[1, "§ 1\xa0Old rubric", 26], [1, "§ 1\xa0New rubric", 30]]
        anchors = ggt.parse_anchors(toc)
        assert anchors["1"] == (30, "New rubric")


class TestDeriveExtents:
    def test_end_page_from_immediately_following_entry(self):
        # §1 and §2 both on page 26 (common -- several norms per page);
        # §1's *next* entry (positionally) is §2, also on page 26.
        toc = [[1, "§ 1\xa0A", 26], [1, "§ 2\xa0B", 26], [1, "§ 3\xa0C", 27]]
        anchors = ggt.parse_anchors(toc)
        norms = ggt.derive_extents(toc, anchors)
        assert norms["1"].start_page == 26
        assert norms["1"].end_page == 26  # next entry same page -> no extra page
        assert norms["2"].end_page == 27  # next entry (§3) one page later

    def test_one_page_gap_yields_two_page_extent(self):
        toc = [[1, "§ 1\xa0A", 26], [1, "§ 2\xa0B", 27]]
        anchors = ggt.parse_anchors(toc)
        norms = ggt.derive_extents(toc, anchors)
        assert norms["1"].start_page == 26
        assert norms["1"].end_page == 27

    def test_large_gap_to_next_norm_is_capped_not_assumed(self):
        # Gap of 14 pages likely means intervening entries were filtered
        # out (e.g. weggefallen norms). The true end is unknown, so the
        # extent is capped at MAX_EXTENT_PAGES rather than claiming the
        # full gap.
        toc = [[1, "§ 1\xa0A", 26], [1, "§ 2\xa0B", 40]]
        anchors = ggt.parse_anchors(toc)
        norms = ggt.derive_extents(toc, anchors)
        assert norms["1"].end_page == 27  # capped, not 39
        assert norms["2"].end_page == 40  # last entry: no next, stays 1-page

    def test_structural_headings_are_skipped_when_finding_next_norm(self):
        # §1's positionally-next TOC entry is a structural "Buch 2"
        # heading far away; the next *norm* entry (§300) determines the
        # extent instead, so §1 isn't penalized for being near a Buch
        # boundary.
        toc = [
            [1, "§ 1\xa0A", 26],
            [1, "Buch 2 Recht der Schuldverhältnisse", 100],
            [1, "§ 300\xa0B", 105],
        ]
        anchors = ggt.parse_anchors(toc)
        norms = ggt.derive_extents(toc, anchors)
        assert norms["1"].end_page == 27  # capped, not 99

    def test_last_norm_uses_own_start_as_end(self):
        toc = [[1, "§ 1\xa0A", 26]]
        anchors = ggt.parse_anchors(toc)
        norms = ggt.derive_extents(toc, anchors)
        assert norms["1"].end_page == 26


class TestValidateAnchor:
    def test_valid_when_citation_and_rubric_present(self):
        norm = ggt.Norm(
            number="1", rubric="Beginn der Rechtsfähigkeit", start_page=26, end_page=26
        )
        text = "§ 1 Beginn der Rechtsfähigkeit des Menschen beginnt mit ..."
        assert ggt.validate_anchor(norm, text) is True

    def test_rejects_missing_citation(self):
        norm = ggt.Norm(
            number="1", rubric="Beginn der Rechtsfähigkeit", start_page=26, end_page=26
        )
        text = "Beginn der Rechtsfähigkeit des Menschen beginnt mit ..."
        skips = ggt.SkipLog()
        assert ggt.validate_anchor(norm, text, skips) is False
        assert skips.counts["citation_not_on_own_page"] == 1

    def test_rejects_missing_rubric_word(self):
        norm = ggt.Norm(
            number="1", rubric="Beginn der Rechtsfähigkeit", start_page=26, end_page=26
        )
        text = "§ 1 hat nichts mit dem Titel zu tun."
        skips = ggt.SkipLog()
        assert ggt.validate_anchor(norm, text, skips) is False
        assert skips.counts["rubric_not_on_own_page"] == 1


class TestHarvestReferrers:
    def test_finds_citation_on_other_page(self):
        page_texts = {1: "hier steht nichts", 2: "hierzu vergleiche § 1 Absatz 2"}
        anchors = {"1": (1, "Test")}
        referrers = ggt.harvest_referrers(page_texts, anchors)
        assert len(referrers["1"]) == 1
        assert referrers["1"][0].page == 2

    def test_excludes_citation_on_own_anchor_page(self):
        page_texts = {1: "§ 1 Test regelt etwas und verweist auf § 1 erneut"}
        anchors = {"1": (1, "Test")}
        referrers = ggt.harvest_referrers(page_texts, anchors)
        assert "1" not in referrers

    def test_ignores_citation_to_unknown_norm(self):
        page_texts = {2: "verweist auf § 999"}
        anchors = {"1": (1, "Test")}
        referrers = ggt.harvest_referrers(page_texts, anchors)
        assert referrers == {}


class TestExtractClause:
    def test_takes_last_sentence_before_citation(self):
        text = "Erster Satz hier. Der Mieter hat ein Widerrufsrecht gemäß § 355."
        idx = text.index("§ 355")
        clause = ggt.extract_clause(text, idx)
        assert clause is not None
        assert "Erster Satz" not in clause
        assert "Widerrufsrecht" in clause

    def test_caps_to_context_window(self):
        long_prefix = " ".join(f"wort{i}" for i in range(40))
        text = f"{long_prefix} § 5"
        idx = text.index("§ 5")
        clause = ggt.extract_clause(text, idx)
        assert clause is not None
        assert len(clause.split()) <= ggt.CONTEXT_WINDOW_TOKENS


class TestStripCitationTokens:
    def test_strips_section_and_digits(self):
        result = ggt.strip_citation_tokens(
            "das Widerrufsrecht gemäß § 355 Absatz 1 Nummer 2"
        )
        assert "§" not in result
        assert not any(ch.isdigit() for ch in result)
        assert "Widerrufsrecht" in result

    def test_removes_empty_parenthetical_shell(self):
        # "(§ 143)" leaves an empty "()" once the citation inside is gone.
        result = ggt.strip_citation_tokens("der Anfechtungsgegner (§ 143) ist")
        assert "(" not in result
        assert ")" not in result
        assert result == "der Anfechtungsgegner ist"

    def test_peels_trailing_dangling_preposition(self):
        # The citation was the object of "nach"; once it's gone, "nach"
        # dangles at the end and should be peeled off too.
        result = ggt.strip_citation_tokens("die Vorschrift des Kaufrechts nach § 445")
        assert result == "die Vorschrift des Kaufrechts"

    def test_peels_multi_word_trailing_phrase(self):
        # "im Sinne des § 312" -> "im Sinne des" all dangle once § 312 is
        # gone and should be peeled iteratively.
        result = ggt.strip_citation_tokens("sind Drittmittel im Sinne des § 312")
        assert result == "sind Drittmittel"


class TestBuildNaturalQuery:
    def test_rejects_clause_that_mostly_restates_the_rubric(self):
        rng = __import__("random").Random(1)
        clause = (
            "das Widerrufsrecht bei außerhalb von Geschäftsräumen "
            "geschlossenen Verträgen"
        )
        result = ggt.build_natural_query(clause, rng, rubric=clause)
        # High token overlap with the target rubric -> rejected as trivial
        assert result is None

    def test_accepts_distinct_clause(self):
        rng = __import__("random").Random(1)
        result = ggt.build_natural_query(
            "der Mieter kann die Wohnung fristlos kündigen wenn Mängel bestehen",
            rng,
            rubric="Kündigung des Mietverhältnisses",
        )
        assert result is not None
        assert "§" not in result
        assert not any(ch.isdigit() for ch in result)

    def test_rejects_too_short_clause(self):
        rng = __import__("random").Random(1)
        result = ggt.build_natural_query("kurzer Satz", rng, rubric="Anderes Thema")
        assert result is None


class TestBuildKeywordQuery:
    def test_strips_trailing_clause_and_caps_tokens(self):
        q = ggt.build_keyword_query("Wohnsitz; Begründung und Aufhebung")
        assert q == "Wohnsitz"

    def test_caps_at_six_tokens(self):
        rubric = "eins zwei drei vier fünf sechs sieben acht"
        q = ggt.build_keyword_query(rubric)
        assert len(q.split()) == 6


class TestStratifyByBuch:
    def test_maps_norms_to_enclosing_buch(self):
        toc = [
            [1, "Buch 1 Allgemeiner Teil", 25],
            [1, "§ 1\xa0A", 26],
            [1, "Buch 2 Recht der Schuldverhältnisse", 100],
            [1, "§ 300\xa0B", 105],
        ]
        anchors = ggt.parse_anchors(toc)
        norms = ggt.derive_extents(toc, anchors)
        mapping = ggt.stratify_by_buch(toc, norms)
        assert mapping["1"] == "Buch 1 Allgemeiner Teil"
        assert mapping["300"] == "Buch 2 Recht der Schuldverhältnisse"


class TestSampleNorms:
    def test_deterministic_for_fixed_seed(self):
        buch_of = {str(i): f"buch{i % 3}" for i in range(30)}
        eligible = [str(i) for i in range(30)]
        s1 = ggt.sample_norms(eligible, buch_of, n=10, seed=42)
        s2 = ggt.sample_norms(eligible, buch_of, n=10, seed=42)
        assert s1 == s2

    def test_different_seeds_differ(self):
        buch_of = {str(i): f"buch{i % 3}" for i in range(30)}
        eligible = [str(i) for i in range(30)]
        s1 = ggt.sample_norms(eligible, buch_of, n=10, seed=1)
        s2 = ggt.sample_norms(eligible, buch_of, n=10, seed=2)
        assert s1 != s2

    def test_force_include_always_present(self):
        buch_of = {str(i): "buch0" for i in range(20)}
        eligible = [str(i) for i in range(20)]
        sampled = ggt.sample_norms(
            eligible, buch_of, n=5, seed=1, force_include=["611a"]
        )
        # force_include not in eligible pool -> filtered out, no crash
        assert "611a" not in sampled
        eligible_with_target = eligible + ["611a"]
        sampled2 = ggt.sample_norms(
            eligible_with_target, buch_of, n=5, seed=1, force_include=["611a"]
        )
        assert "611a" in sampled2

    def test_does_not_exceed_n(self):
        buch_of = {str(i): f"buch{i % 4}" for i in range(50)}
        eligible = [str(i) for i in range(50)]
        sampled = ggt.sample_norms(eligible, buch_of, n=12, seed=7)
        assert len(sampled) == 12


class TestBuildScenarios:
    def test_produces_matched_k_and_n_pair(self):
        norms = {
            "355": ggt.Norm(
                number="355", rubric="Widerrufsrecht", start_page=82, end_page=82
            )
        }
        referrers = {
            "355": [
                ggt.Referrer(page=88, norm="355", char_start=40, char_end=45),
            ]
        }
        page_texts = {
            88: (
                "Der Verbraucher hat bei außerhalb von Geschäftsräumen "
                "geschlossenen Verträgen ein Widerrufsrecht gemäß § 355."
            )
        }
        skips = ggt.SkipLog()
        scenarios = ggt.build_scenarios(
            ["355"], norms, referrers, page_texts, seed=1, skips=skips
        )
        assert "de01k" in scenarios
        assert "de01n" in scenarios
        assert scenarios["de01k"]["relevant_pages"] == [82]
        assert scenarios["de01n"]["relevant_pages"] == [82]
        assert scenarios["de01k"]["arm"] == "keyword_control"
        assert scenarios["de01n"]["arm"] == "semantic_xref"
        assert "§" not in scenarios["de01n"]["query"]

    def test_falls_back_to_keyword_only_when_no_referrer(self):
        # e.g. a force-included norm (like § 611a) that no other page
        # cites: still emit the keyword_control arm rather than dropping
        # the norm from the output entirely.
        norms = {"1": ggt.Norm(number="1", rubric="X", start_page=1, end_page=1)}
        skips = ggt.SkipLog()
        scenarios = ggt.build_scenarios(["1"], norms, {}, {}, seed=1, skips=skips)
        assert list(scenarios.keys()) == ["de01k"]
        assert scenarios["de01k"]["arm"] == "keyword_control"
        assert skips.counts["no_referrer"] == 1

    def test_falls_back_to_keyword_only_when_referrer_too_close(self):
        norms = {"1": ggt.Norm(number="1", rubric="X", start_page=10, end_page=10)}
        referrers = {"1": [ggt.Referrer(page=11, norm="1", char_start=0, char_end=5)]}
        skips = ggt.SkipLog()
        scenarios = ggt.build_scenarios(
            ["1"], norms, referrers, {11: "text"}, seed=1, skips=skips
        )
        assert list(scenarios.keys()) == ["de01k"]
        assert skips.counts["referrer_too_close"] == 1


class TestGroundTruthShapeCompatibility:
    """The generated JSON must be directly consumable by
    scripts/benchmark_embedding_models.py's existing scenario_k / arm
    handling (added in this same change)."""

    def test_scenarios_carry_required_keys(self):
        norms = {
            "355": ggt.Norm(
                number="355", rubric="Widerrufsrecht", start_page=82, end_page=82
            )
        }
        referrers = {
            "355": [ggt.Referrer(page=88, norm="355", char_start=40, char_end=45)]
        }
        page_texts = {
            88: (
                "Der Verbraucher hat bei außerhalb von Geschäftsräumen "
                "geschlossenen Verträgen ein Widerrufsrecht gemäß § 355."
            )
        }
        skips = ggt.SkipLog()
        scenarios = ggt.build_scenarios(
            ["355"], norms, referrers, page_texts, seed=1, skips=skips
        )
        for sid, s in scenarios.items():
            assert "query" in s
            assert "relevant_pages" in s
            assert isinstance(s["relevant_pages"], list)
            assert "k" in s
            assert "arm" in s
