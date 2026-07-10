from __future__ import annotations



from iai_mcp.pipeline import _trigram_jaccard

class TestDeterministicBoost:
    def test_trigram_jaccard_exact_match(self):
        assert _trigram_jaccard("abc123def456", "abc123def456") == 1.0

    def test_trigram_jaccard_partial_overlap(self):
        score = _trigram_jaccard("abc123", "abc123def456")
        assert score > 0.3

    def test_trigram_jaccard_no_overlap(self):
        score = _trigram_jaccard("zzzzz", "yyyyy")
        assert score == 0.0

    def test_trigram_jaccard_short_strings(self):
        assert _trigram_jaccard("ab", "abc") == 0.0

    def test_exact_substring_boost_logic(self):
        cue = "abc123def456"
        surfaces = {
            "rec1": "The token is abc123def456 for prod",
            "rec2": "Some unrelated memory about python",
            "rec3": "Another record with abc123def456 inside",
        }
        cue_lower = cue.lower()
        fts_hits = set()
        for rid, surface in surfaces.items():
            if cue_lower in surface.lower():
                fts_hits.add(rid)
        assert "rec1" in fts_hits
        assert "rec3" in fts_hits
        assert "rec2" not in fts_hits

    def test_exact_substring_boost_case_insensitive(self):
        cue = "ABC123DEF"
        surface = "the code is abc123def in this context"
        assert cue.lower() in surface.lower()

    def test_exact_substring_boost_minimum_length(self):
        cue = "ab"
        assert len(cue) < 4

    def test_hex_string_found_by_substring(self):
        cue = "7695c69f8a4b"
        surface = "Commit SHA: 7695c69f8a4b2c1d on main branch"
        assert cue.lower() in surface.lower()
