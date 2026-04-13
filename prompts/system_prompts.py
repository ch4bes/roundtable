from dataclasses import dataclass


@dataclass
class ModeratorPrompt:
    @staticmethod
    def system() -> str:
        return (
            "You are a neutral discussion moderator. Extract key points. Output bullet points only."
        )

    @staticmethod
    def template(responses: list[dict[str, str]], round_num: int) -> str:
        response_text = "\n".join(f"{r['model']}: {r['content']}" for r in responses)
        return f"Round {round_num} responses:\n{response_text}\n\nBullet points:"


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
    def with_context(
        prompt: str,
        context_responses: list[dict[str, str]],
        model_position: int,
        total_models: int,
        round_num: int,
    ) -> str:
        context_text = "\n".join(f"{r['model']}: {r['content']}" for r in context_responses)
        return f"{prompt}\n\nRESPONSES:\n{context_text}\n\nModel {model_position}/{total_models} round {round_num}."
