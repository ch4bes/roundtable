from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from storage.session import AttributedSummary


@dataclass
class ModeratorPrompt:
    @staticmethod
    def system(threshold: float = 0.75) -> str:
        return f"""You are a neutral discussion moderator. Your task is to:
1. Summarize each participant's main points (attributed to them)
2. Analyze ALL clusters in the similarity matrix to determine agreement
3. Identify the MAIN QUESTION being asked and determine if participants agree on the MAIN ANSWER
4. Provide a SINGLE definitive consensus assessment

CRITICAL RULES:
- Analyze the FULL similarity matrix (ALL pairs, not just high-similarity ones) before forming any conclusion
- Use CONSENSUS THRESHOLD of {threshold}: pairs above {threshold} indicate strong agreement, below indicate different perspectives
- Identify ALL clusters in the matrix: scores >{threshold} are strongly aligned, {threshold-0.3}-{threshold} are somewhat aligned, <{threshold-0.3} are different perspectives
- For each cluster, determine what position they hold on the MAIN QUESTION
- Determine if ALL clusters agree on the MAIN ANSWER, even if they provide different EXAMPLES

CLUSTER ANALYSIS:
- Group participants by similarity: high-similarity pairs (>{threshold}) form clusters
- Identify each cluster's core position on the main question
- Check: Do ALL clusters agree on the MAIN ANSWER?
- Different examples/supporting evidence = PERIPHERAL disagreement (does NOT prevent consensus)
- Different core positions on main question = FUNDAMENTAL disagreement (prevents consensus)

MAIN QUESTION vs MAIN ANSWER vs PERIPHERAL:
- MAIN QUESTION: The core topic being debated (e.g., "Is there a best movie?")
- MAIN ANSWER: The core position on that question (e.g., "No, it's subjective")
- PERIPHERAL: Supporting details, examples, or specific choices (e.g., "Godfather vs Kane")

CONSENSUS RULE:
- If ALL clusters agree on the MAIN ANSWER → Consensus: REACHED
- If clusters fundamentally disagree on MAIN ANSWER → Consensus: NOT REACHED
- Peripheral disagreements (different examples) do NOT prevent consensus on the main answer

Output format EXACTLY:

## Individual Summaries

### MODEL_NAME
- Point 1
- Point 2

### MODEL_NAME
- Point 1
- Point 2

## Similarity Matrix Analysis
- List ALL clusters identified (high-similarity groups)
- For each cluster: which participants, what position they hold
- Check if all clusters agree on the main answer despite different examples

## Agreement Analysis
- Areas of full agreement (reference specific responses)
- Areas of partial agreement (reference specific responses)  
- Areas of disagreement (reference specific responses)
- Reference the similarity matrix when discussing agreement strength

## Final Consensus
Consensus: REACHED / NOT REACHED
Confidence: HIGH / MEDIUM / LOW
Justification: One sentence explaining why the main answer is or is not agreed upon
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
        threshold: float = 0.75,
    ) -> str:
        response_text = "\n\n".join(f"### {r['model']}\n{r['content']}" for r in responses)

        matrix_table = ModeratorPrompt._format_similarity_matrix(similarity_matrix, model_names)

        return f"""Round {round_num} responses:

{response_text}

=== SIMILARITY MATRIX ===
Pairwise similarities between responses (embedding-based):

{matrix_table}

ANALYSIS GUIDELINES:
- Pairs above {threshold} = strong agreement (same cluster)
- Pairs {threshold-0.3} to {threshold} = somewhat aligned (different but related)
- Pairs below {threshold-0.3} = different perspectives
- Identify ALL clusters, not just the high-similarity ones
- Check if all clusters agree on the main answer despite peripheral disagreements

Follow the format. Use the similarity matrix to identify clusters and determine consensus. Place your ONLY Consensus verdict in ## Final Consensus at the end."""

    @staticmethod
    def final_review(
        prompt: str,
        all_responses: list,
        all_summaries: list,
    ) -> tuple[str, str]:
        system = """You are writing the final review of a completed roundtable discussion.
Your task is to provide a comprehensive analysis of the entire discussion.

STRUCTURE YOUR RESPONSE:

## Discussion Overview
- Main question debated
- Total rounds completed
- Participants involved

## Key Arguments by Position
[Group arguments by stance, not by model - show evolution across rounds]

## Points of Agreement
- Universal consensus points
- Majority agreement points

## Points of Disagreement
- Fundamental disagreements (prevented consensus)
- Peripheral disagreements (different examples/evidence)

## Position Evolution
[Natural language narrative of how positions shifted]
- Which models changed positions and when
- What counterarguments influenced changes
- Critical turning points in the discussion

## Final Assessment
- Consensus: REACHED/NOT REACHED
- If reached: the agreed position statement
- If not reached: what remains unresolved
- Recommendation for further discussion

Be comprehensive but concise. Reference specific rounds and models when relevant."""

        rounds_data: dict[int, list] = {}
        for resp in all_responses:
            r = resp.round
            if r not in rounds_data:
                rounds_data[r] = []
            rounds_data[r].append(resp)

        discussion_text = ""
        for round_num_key in sorted(rounds_data.keys()):
            discussion_text += f"### Round {round_num_key}\n\n"
            for resp in rounds_data[round_num_key]:
                discussion_text += f"**{resp.model}:**\n{resp.content}\n\n"

        summaries_text = ""
        for summary in all_summaries:
            summaries_text += (
                f"### Round {summary.round} Analysis\n"
                f"Consensus: {summary.consensus_assessment} (Confidence: {summary.confidence})\n"
                f"Agreement: {summary.agreement_analysis}\n\n"
            )

        user_prompt = f"""Original Prompt: {prompt}

=== FULL DISCUSSION HISTORY ===

{discussion_text}
=== ROUND SUMMARIES ===

{summaries_text}
Generate the final review following the structure above."""

        return (system, user_prompt)


@dataclass
class ParticipantPrompt:
    @staticmethod
    def system() -> str:
        return "You are a roundtable participant. Provide direct answers. Avoid greetings, hedging, and meta-commentary."

    @staticmethod
    def initial(prompt: str) -> str:
        return f"{prompt}\n\nProvide your direct answer in 2-4 paragraphs."

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

RESPONSE OPTIONS:
- REFINEMENT: Strengthen your previous position with new reasoning
- REVISION: Abandon flawed arguments and adopt a more convincing position
- REINFORCEMENT: Support consensus points while adding unique insights
- DISAGREEMENT: Clearly state what you disagree with and justify why

Be direct about which path you're taking and why.

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
