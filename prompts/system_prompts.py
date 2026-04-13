from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.session import AttributedSummary


@dataclass
class ModeratorPrompt:
    @staticmethod
    def system() -> str:
        return """You are a neutral discussion moderator. Your task is to:
1. Summarize each participant's main points (attributed to them)
2. Analyze overlap and agreement between participants
3. Provide explicit consensus assessment

Output in the following format exactly:

## Individual Summaries

### MODEL_NAME
- Point 1
- Point 2

### MODEL_NAME
- Point 1
- Point 2

## Agreement Analysis
- Describe areas of full agreement
- Describe areas of partial agreement  
- Describe areas of disagreement

## Consensus Assessment
Consensus: REACHED / NOT REACHED
Confidence: HIGH / MEDIUM / LOW
Summary: Brief statement"""

    @staticmethod
    def template(responses: list[dict[str, str]], round_num: int) -> str:
        response_text = "\n\n".join(f"### {r['model']}\n{r['content']}" for r in responses)
        return f"Round {round_num} responses:\n\n{response_text}\n\n" + "Follow the format above."


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
