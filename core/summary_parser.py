"""
SummaryParser: extracts structured fields from moderator-generated text.

Decoupled from DiscussionOrchestrator so it can be tested independently and
reused by any component that needs to interpret a moderator response.

Parsing strategy (two-tier):
  1. JSON block (preferred) — reliable when the model honours the format.
  2. Regex Markdown fallback — tolerant of LLM formatting variations.
"""

import json
import logging
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from storage.session import Response


class SummaryParser:
    """Static helpers for parsing moderator attributed-summary output."""

    @staticmethod
    def parse(text: str, round_responses: "list[Response]") -> dict:
        """Parse a moderator summary into structured fields.

        Tries JSON extraction first; falls back to Markdown regex parsing.

        Returns a dict with keys:
            individual_summaries: dict[str, list[str]]
            agreement_analysis: str
            consensus_assessment: "REACHED" | "NOT REACHED"
            confidence: "HIGH" | "MEDIUM" | "LOW"
        """
        parsed = SummaryParser._parse_json_block(text)
        if parsed is not None:
            return parsed
        return SummaryParser._parse_markdown(text, round_responses)

    # ------------------------------------------------------------------
    # Tier 1 — JSON block
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_block(text: str) -> dict | None:
        """Try to extract a JSON block from fenced code (```json ... ```)."""
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if (
                    "individual_summaries" in parsed
                    and "consensus_assessment" in parsed
                ):
                    return {
                        "individual_summaries": parsed.get("individual_summaries", {}),
                        "agreement_analysis": parsed.get(
                            "agreement_analysis", "(Not provided)"
                        ),
                        "consensus_assessment": parsed.get(
                            "consensus_assessment", "NOT REACHED"
                        ),
                        "confidence": parsed.get("confidence", "MEDIUM"),
                    }
            except json.JSONDecodeError:
                pass
        return None

    # ------------------------------------------------------------------
    # Tier 2 — Regex Markdown fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_markdown(text: str, round_responses: "list[Response]") -> dict:
        """Regex-based Markdown parsing, tolerant of LLM formatting variations."""
        individual_summaries: dict[str, list[str]] = {}
        agreement_analysis = ""
        consensus_assessment = "NOT REACHED"
        confidence = "MEDIUM"

        lines = text.split("\n")
        current_model: str | None = None
        current_section: str | None = None

        for line in lines:
            stripped = line.strip()

            # HEADINGS: match 2-5 # with optional spacing
            heading_match = re.match(r"^#{2,5}\s?(.*)", stripped)
            if heading_match:
                heading_text = heading_match.group(1).strip()
                heading_level = len(heading_match.group(0)) - len(
                    heading_match.group(0).lstrip("#")
                )

                # 3+ hashes (###) → individual model section
                if heading_level >= 3 and heading_text:
                    current_model = heading_text
                    individual_summaries[current_model] = []
                    current_section = "individual"

                # 2 hashes (##) → major section headings
                elif heading_level == 2 and heading_text:
                    current_model = None
                    heading_lower = heading_text.lower()

                    if "final" in heading_lower and "consensus" in heading_lower:
                        current_section = "final_consensus"
                    elif "agreement" in heading_lower:
                        current_section = "agreement"
                    elif "similarity" in heading_lower:
                        current_section = "similarity"
                    elif "individual" in heading_lower and "summary" in heading_lower:
                        current_section = "individual_summaries"
                    else:
                        current_section = "other"
                else:
                    current_section = None
                    current_model = None

                continue

            # BULLETS inside a model section
            if current_model:
                bullet_match = re.match(r"^[-*•–—]\s+(.*)", stripped)
                if bullet_match:
                    individual_summaries[current_model].append(
                        bullet_match.group(1).strip()
                    )
                    continue

            # CONSENSUS VERDICT — prefer Final Consensus section
            if current_section == "final_consensus":
                if re.search(r"consensus\s*:\s*", stripped, re.I):
                    if "NOT REACHED" in stripped.upper():
                        consensus_assessment = "NOT REACHED"
                    elif "REACHED" in stripped.upper():
                        consensus_assessment = "REACHED"
            elif (
                current_section != "final_consensus"
                and consensus_assessment == "NOT REACHED"
            ):
                if re.search(r"consensus\s+assessment\s*:\s*", stripped, re.I):
                    if "NOT REACHED" in stripped.upper():
                        consensus_assessment = "NOT REACHED"
                    elif "REACHED" in stripped.upper():
                        consensus_assessment = "REACHED"

            # CONFIDENCE
            if re.search(r"confidence\s*:|CONFIDENCE:", stripped, re.I):
                if "HIGH" in stripped.upper():
                    confidence = "HIGH"
                elif "LOW" in stripped.upper():
                    confidence = "LOW"
                else:
                    confidence = "MEDIUM"

            # AGREEMENT ANALYSIS
            if (
                current_section == "agreement"
                and stripped
                and not stripped.startswith("#")
            ):
                if not agreement_analysis:
                    agreement_analysis = stripped
                else:
                    agreement_analysis += "\n" + stripped

        if not individual_summaries:
            for r in round_responses:
                individual_summaries[r.model] = ["(No specific points extracted)"]

        if not agreement_analysis:
            agreement_analysis = "(Analysis not provided)"

        # Guard against contradiction: agreement_analysis says NOT REACHED but
        # consensus_assessment was set to REACHED.
        if consensus_assessment == "REACHED" and agreement_analysis:
            agreement_upper = agreement_analysis.upper()
            if (
                re.search(r"consensus\s*:\s*not\s*reached", agreement_upper)
                or re.search(r"consensus\s+not\s+reached", agreement_upper)
                or re.search(r"no\s+consensus", agreement_upper)
            ):
                logger.warning(
                    "Consensus assessment contradiction: "
                    "assessment=%s but agreement_analysis contains 'NOT REACHED'. "
                    "Defaulting to NOT REACHED.",
                    consensus_assessment,
                )
                consensus_assessment = "NOT REACHED"

        return {
            "individual_summaries": individual_summaries,
            "agreement_analysis": agreement_analysis,
            "consensus_assessment": consensus_assessment,
            "confidence": confidence,
        }
