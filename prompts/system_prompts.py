from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from storage.session import AttributedSummary


@dataclass
class ModeratorPrompt:
    @staticmethod
    def system() -> str:
        return """You are a neutral discussion moderator. Your task is to:
1. Summarize each participant's main points (attributed to them)
2. Analyze overlap and agreement between participants using the full data provided
3. Provide a SINGLE definitive consensus assessment

CRITICAL RULES:
- Analyze ALL responses and the similarity matrix FULLY before forming any conclusion
- Output EXACTLY ONE "Consensus" statement, placed ONLY in the "## Final Consensus" section
- NEVER output a consensus assessment, verdict, or summary conclusion before the final section
- Verify your conclusion matches your analysis. If your analysis finds disagreement, your conclusion must say NOT REACHED
- The Agreement Analysis is where you reason through the data. The Final Consensus is where you state the result.

Output format EXACTLY:

## Individual Summaries

### MODEL_NAME
- Point 1
- Point 2

### MODEL_NAME
- Point 1
- Point 2

## Agreement Analysis
- Areas of full agreement (reference specific responses)
- Areas of partial agreement (reference specific responses)
- Areas of disagreement (reference specific responses)
- Reference the similarity matrix when discussing agreement strength

## Final Consensus
Consensus: REACHED / NOT REACHED
Confidence: HIGH / MEDIUM / LOW
Justification: One sentence linking your conclusion to the analysis above
"""

    @staticmethod
    def template(responses: list[dict[str, str]], round_num: int) -> str:
        response_text = "\n\n".join(f"### {r['model']}\n{r['content']}" for r in responses)
        return f"Round {round_num} responses:\n\n{response_text}\n\n" + "Follow the format above."

    @staticmethod
    def _format_similarity_matrix(matrix: np.ndarray, model_names: list[str]) -> str:
        n = matrix.shape[0]
        if n == 0:
            return "(No responses to compare)"

        short_names = []
        for name in model_names:
            parts = name.split(":")
            if len(parts) >= 2:
                short = f"{parts[0].replace('qwen', 'q')}{parts[1][:3]}"
                short_names.append(short)
            else:
                short_names.append(name[:8])

        max_name_len = max(len(name) for name in short_names)
        col_width = max(8, max_name_len + 2)

        header = "| " + " | ".join(name.center(col_width) for name in short_names) + " |"
        separator = "|" + "|".join("-" * (col_width + 2) for _ in range(n + 1)) + "|"

        rows = []
        for i in range(n):
            row_values = [short_names[i].ljust(max_name_len)]
            for j in range(n):
                if i == j:
                    row_values.append(f"{matrix[i, j]:.2f}".center(col_width))
                elif i < j:
                    row_values.append(f"{matrix[i, j]:.2f}".center(col_width))
                else:
                    row_values.append(" " * col_width)
            rows.append("| " + " | ".join(row_values) + " |")

        return "\n".join([header, separator] + rows)

    @staticmethod
    def template_with_similarity_matrix(
        responses: list[dict[str, str]],
        round_num: int,
        similarity_matrix: np.ndarray,
        model_names: list[str],
    ) -> str:
        response_text = "\n\n".join(f"### {r['model']}\n{r['content']}" for r in responses)

        matrix_table = ModeratorPrompt._format_similarity_matrix(similarity_matrix, model_names)

        return f"""Round {round_num} responses:

{response_text}

=== SIMILARITY MATRIX ===
Pairwise similarities between responses (embedding-based):

{matrix_table}

Use this data to inform your Agreement Analysis. High values (>0.7) indicate strong agreement; low values (<0.4) indicate significant disagreement.

Follow the format. Use the similarity matrix in your Analysis. Place your ONLY Consensus verdict in ## Final Consensus at the end.
"""


@dataclass
class ParticipantPrompt:
    @staticmethod
    def system() -> str:
        return "You are a roundtable participant. Provide direct answers. Avoid greetings, hedging, and meta-commentary."

    @staticmethod
    def initial(prompt: str, model_position: int, total_models: int) -> str:
        return (
            f"{prompt}\n\nModel {model_position}/{total_models}. Direct answer in 2-4 paragraphs."
        )

    @staticmethod
    def with_summary(
        prompt: str, summary: str, model_position: int, total_models: int, round_num: int
    ) -> str:
        return f"{prompt}\n\nSUMMARY:\n{summary}\n\nModel {model_position}/{total_models} round {round_num}. Updated response."

    @staticmethod
    def with_attributed_summary(
        prompt: str,
        attributed: "AttributedSummary",
        model_position: int,
        total_models: int,
        round_num: int,
    ) -> str:
        individual_text = "\n\n".join(
            f"### {model}\n" + "\n".join(f"- {p}" for p in points)
            for model, points in attributed.individual_summaries.items()
        )

        return f"""{prompt}

PREVIOUS ROUND DISCUSSION:

{individual_text}

AGREEMENT ANALYSIS:
{attributed.agreement_analysis}

CONSENSUS ASSESSMENT: {attributed.consensus_assessment} (Confidence: {attributed.confidence})

Model {model_position}/{total_models} round {round_num}.
Consider the above analysis. If you agree with the consensus, reinforce the key points.
If you disagree or have additional insights, clearly state what you disagree with and why.
Provide your updated position."""

    @staticmethod
    def with_context(
        prompt: str,
        context_responses: list[dict[str, str]],
        model_position: int,
        total_models: int,
        round_num: int,
    ) -> str:
        context_text = "\n".join(f"{r['model']}: {r['content']}" for r in context_responses)
        return f"{prompt}\n\nRESPONSES:\n{context_text}\n\nModel {model_position}/{total_models} round {round_num}."
