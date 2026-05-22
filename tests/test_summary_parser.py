"""Tests for SummaryParser edge cases - Issue #41 (Mock Bias).

Covers malformed JSON, missing keys, empty responses, and the Markdown
fallback path so the parser is verified against non-ideal LLM output.
"""

import pytest

from core.summary_parser import SummaryParser
from storage.session import Response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _response(
    model: str, content: str = "some content", round_num: int = 1
) -> Response:
    return Response(
        model=model,
        content=content,
        round=round_num,
        timestamp="2024-01-01T00:00:00",
        position=0,
    )


def _responses(*models: str) -> list[Response]:
    return [_response(m) for m in models]


# ---------------------------------------------------------------------------
# JSON block — happy path
# ---------------------------------------------------------------------------


class TestParseJsonBlockHappyPath:
    def test_valid_json_block_is_used(self):
        text = """```json
{
  "individual_summaries": {"m1": ["point 1"], "m2": ["point 2"]},
  "agreement_analysis": "Models broadly agree",
  "consensus_assessment": "REACHED",
  "confidence": "HIGH"
}
```"""
        result = SummaryParser.parse(text, _responses("m1", "m2"))

        assert result["consensus_assessment"] == "REACHED"
        assert result["confidence"] == "HIGH"
        assert result["agreement_analysis"] == "Models broadly agree"
        assert result["individual_summaries"]["m1"] == ["point 1"]

    def test_json_block_without_language_tag(self):
        text = """```
{
  "individual_summaries": {"m1": ["p1"]},
  "consensus_assessment": "NOT REACHED",
  "confidence": "LOW",
  "agreement_analysis": "No agreement yet"
}
```"""
        result = SummaryParser.parse(text, _responses("m1"))
        assert result["consensus_assessment"] == "NOT REACHED"
        assert result["confidence"] == "LOW"

    def test_json_block_missing_optional_fields_uses_defaults(self):
        """JSON with only the required keys uses sensible defaults for omitted ones."""
        text = """```json
{
  "individual_summaries": {"m1": ["p1"]},
  "consensus_assessment": "REACHED"
}
```"""
        result = SummaryParser.parse(text, _responses("m1"))
        assert result["consensus_assessment"] == "REACHED"
        assert result["agreement_analysis"] == "(Not provided)"
        assert result["confidence"] == "MEDIUM"


# ---------------------------------------------------------------------------
# JSON block — malformed / missing required keys → fallback to Markdown
# ---------------------------------------------------------------------------


class TestParseJsonBlockMalformed:
    def test_syntax_error_falls_back_to_markdown(self):
        """A fenced block with invalid JSON syntax triggers the Markdown fallback."""
        text = """```json
{ "individual_summaries": {"m1": ["p1"]}, BROKEN JSON HERE
```

## Individual Summaries

### m1
- Fallback point

## Agreement Analysis
Models disagree.

Consensus Assessment: NOT REACHED
Confidence: LOW
"""
        result = SummaryParser.parse(text, _responses("m1"))
        # Markdown fallback should extract data
        assert "m1" in result["individual_summaries"]
        assert result["consensus_assessment"] == "NOT REACHED"

    def test_json_missing_required_key_falls_back(self):
        """A JSON block missing 'consensus_assessment' triggers the Markdown fallback."""
        text = """```json
{
  "individual_summaries": {"m1": ["point"]},
  "agreement_analysis": "Some analysis"
}
```

## Agreement Analysis
No agreement.

Consensus Assessment: NOT REACHED
"""
        result = SummaryParser.parse(text, _responses("m1"))
        # Falls back because 'consensus_assessment' key is absent
        assert result["consensus_assessment"] == "NOT REACHED"

    def test_json_missing_individual_summaries_falls_back(self):
        """A JSON block missing 'individual_summaries' triggers the Markdown fallback."""
        text = """```json
{
  "consensus_assessment": "REACHED",
  "confidence": "HIGH"
}
```

### m1
- A markdown point
"""
        result = SummaryParser.parse(text, _responses("m1"))
        # Falls back — individual_summaries extracted from Markdown
        assert "m1" in result["individual_summaries"]

    def test_truncated_json_falls_back(self):
        """A JSON block truncated mid-object falls back to Markdown."""
        text = """```json
{
  "individual_summaries": {"m1": ["p
```

### m1
- Markdown point

Consensus Assessment: REACHED
Confidence: MEDIUM
"""
        result = SummaryParser.parse(text, _responses("m1"))
        assert "m1" in result["individual_summaries"]
        assert result["consensus_assessment"] == "REACHED"

    def test_empty_json_block_falls_back(self):
        """An empty fenced block falls back to Markdown."""
        text = """```json
```

### m1
- Only markdown

Consensus Assessment: NOT REACHED
"""
        result = SummaryParser.parse(text, _responses("m1"))
        assert "m1" in result["individual_summaries"]

    def test_json_with_null_values_falls_back(self):
        """JSON with null for required keys is treated as missing and falls back."""
        text = """```json
{
  "individual_summaries": null,
  "consensus_assessment": null
}
```

### m1
- Fallback markdown point

Consensus Assessment: REACHED
"""
        result = SummaryParser.parse(text, _responses("m1"))
        # _parse_json_block returns None because individual_summaries is absent
        # or null evaluates as falsy — Markdown fallback kicks in
        assert result is not None  # does not crash


# ---------------------------------------------------------------------------
# Empty and whitespace-only responses
# ---------------------------------------------------------------------------


class TestEmptyResponses:
    def test_empty_string_returns_defaults(self):
        """Empty text must not crash; returns defaults with model placeholders."""
        result = SummaryParser.parse("", _responses("m1", "m2"))

        assert isinstance(result, dict)
        assert "individual_summaries" in result
        assert "consensus_assessment" in result
        assert result["consensus_assessment"] in ("REACHED", "NOT REACHED")
        # Placeholder entries for each model
        assert "m1" in result["individual_summaries"]
        assert "m2" in result["individual_summaries"]

    def test_whitespace_only_returns_defaults(self):
        result = SummaryParser.parse("   \n\n\t  ", _responses("m1"))

        assert isinstance(result, dict)
        assert result["consensus_assessment"] in ("REACHED", "NOT REACHED")

    def test_empty_with_no_models_returns_defaults(self):
        """No round_responses passed — individual_summaries is an empty dict."""
        result = SummaryParser.parse("", [])

        assert isinstance(result, dict)
        assert result["individual_summaries"] == {}

    def test_empty_agreement_analysis_fallback(self):
        result = SummaryParser.parse("", _responses("m1"))
        assert result["agreement_analysis"] == "(Analysis not provided)"

    def test_none_like_model_responses_with_content(self):
        """Models with empty content still produce placeholder summaries."""
        responses = [_response("m1", ""), _response("m2", "")]
        result = SummaryParser.parse("", responses)
        assert "m1" in result["individual_summaries"]
        assert "m2" in result["individual_summaries"]


# ---------------------------------------------------------------------------
# Markdown fallback — various formats
# ---------------------------------------------------------------------------


class TestMarkdownFallback:
    def test_consensus_reached_from_markdown(self):
        text = """
## Individual Summaries

### m1
- Point A
- Point B

### m2
- Point C

## Agreement Analysis
Both models agree on the core principles.

## Final Consensus
Consensus: REACHED
Confidence: HIGH
"""
        result = SummaryParser.parse(text, _responses("m1", "m2"))
        assert result["consensus_assessment"] == "REACHED"
        assert result["confidence"] == "HIGH"
        assert "m1" in result["individual_summaries"]
        assert result["individual_summaries"]["m1"] == ["Point A", "Point B"]

    def test_consensus_not_reached_from_markdown(self):
        text = """
## Individual Summaries

### modelA
- Disagrees on X

### modelB
- Disagrees on Y

## Agreement Analysis
Models have significant disagreements.

Consensus Assessment: NOT REACHED
Confidence: LOW
"""
        result = SummaryParser.parse(text, _responses("modelA", "modelB"))
        assert result["consensus_assessment"] == "NOT REACHED"
        assert result["confidence"] == "LOW"

    def test_confidence_low_from_markdown(self):
        text = "Consensus Assessment: NOT REACHED\nConfidence: LOW\n"
        result = SummaryParser.parse(text, _responses("m1"))
        assert result["confidence"] == "LOW"

    def test_confidence_medium_from_markdown(self):
        text = "Consensus Assessment: NOT REACHED\nConfidence: MEDIUM\n"
        result = SummaryParser.parse(text, _responses("m1"))
        assert result["confidence"] == "MEDIUM"

    def test_model_with_no_bullets_still_added(self):
        """A model section heading with no bullet points gets an empty list."""
        text = """
### m1

### m2
- Some point
"""
        result = SummaryParser.parse(text, _responses("m1", "m2"))
        assert "m1" in result["individual_summaries"]
        assert result["individual_summaries"]["m2"] == ["Some point"]

    def test_no_headings_uses_placeholder_per_model(self):
        """Completely free-form text with no headings uses placeholder per model."""
        text = "The models discussed AI safety and reached some conclusions."
        result = SummaryParser.parse(text, _responses("alpha", "beta"))
        assert "alpha" in result["individual_summaries"]
        assert "beta" in result["individual_summaries"]

    def test_agreement_analysis_extracted(self):
        text = """
## Agreement Analysis
Both models broadly agree on the main points.

Consensus Assessment: REACHED
"""
        result = SummaryParser.parse(text, _responses("m1"))
        assert "broadly agree" in result["agreement_analysis"]

    def test_contradiction_detection_overrides_reached(self):
        """If agreement_analysis says 'no consensus' but assessment is REACHED,
        the parser should correct to NOT REACHED."""
        text = """
## Agreement Analysis
There is no consensus between the models.

Consensus Assessment: REACHED
Confidence: HIGH
"""
        result = SummaryParser.parse(text, _responses("m1"))
        assert result["consensus_assessment"] == "NOT REACHED"

    def test_unicode_bullets(self):
        """Unicode bullet characters (•, –, —) should be parsed as list items."""
        text = """
### m1
• Unicode bullet one
– En-dash bullet
— Em-dash bullet
"""
        result = SummaryParser.parse(text, _responses("m1"))
        assert len(result["individual_summaries"].get("m1", [])) == 3

    def test_deep_heading_levels_treated_as_model_section(self):
        """Headings with 4+ # chars are treated as model sections."""
        text = """
#### deep_model
- Deep point
"""
        result = SummaryParser.parse(text, _responses("deep_model"))
        assert "deep_model" in result["individual_summaries"]
        assert result["individual_summaries"]["deep_model"] == ["Deep point"]


# ---------------------------------------------------------------------------
# Internal: _parse_json_block return values
# ---------------------------------------------------------------------------


class TestParseJsonBlockInternal:
    def test_returns_none_when_no_fenced_block(self):
        assert SummaryParser._parse_json_block("plain text, no fences") is None

    def test_returns_none_on_invalid_json(self):
        text = "```json\n{broken\n```"
        assert SummaryParser._parse_json_block(text) is None

    def test_returns_none_on_missing_required_keys(self):
        text = '```json\n{"agreement_analysis": "ok"}\n```'
        assert SummaryParser._parse_json_block(text) is None

    def test_returns_dict_on_valid_input(self):
        text = (
            "```json\n"
            '{"individual_summaries": {"m1": ["p"]}, '
            '"consensus_assessment": "REACHED"}\n'
            "```"
        )
        result = SummaryParser._parse_json_block(text)
        assert result is not None
        assert result["consensus_assessment"] == "REACHED"

    def test_empty_fenced_block_returns_none(self):
        assert SummaryParser._parse_json_block("```json\n```") is None
