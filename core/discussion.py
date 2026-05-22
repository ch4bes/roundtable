import asyncio
import json
import random
import re
import shutil
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from prompts.system_prompts import ModeratorPrompt, ParticipantPrompt
from storage.session import Response, Session, SessionManager

from .config import Config
from .consensus import ConsensusDetector, ConsensusResult
from .input_reader import InputBuffer
from .llm_client import LLMClient
from .ollama_client import OllamaClient
from .similarity import SimilarityEngine
from .summary_parser import SummaryParser
from .tools import create_tool_executor, get_available_tools


class _ConsensusVerdict:
    REACHED = "REACHED"
    NOT_REACHED = "NOT_REACHED"
    INCONSISTENT = "INCONSISTENT"


@dataclass
class DiscussionState:
    current_round: int
    current_model_index: int
    model_order: list[str]
    consensus_result: ConsensusResult | None = None
    is_running: bool = False
    is_paused: bool = False
    skip_requested: bool = False


class DiscussionOrchestrator:
    def __init__(
        self,
        config: Config,
        session: Session,
        progress_callback: Callable[[DiscussionState], Awaitable[None]] | None = None,
        human_input_callback: Callable[[str, int, int], Awaitable[str]] | None = None,
        input_buffer: InputBuffer | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.config = config
        self.session = session
        self.human_input_callback = human_input_callback
        self.input_buffer = input_buffer
        # Accept an injected client (enables testing with mocks and alternative
        # backends); fall back to OllamaClient when none is provided.
        self.ollama: LLMClient = llm_client or OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
        )
        self.similarity_engine = SimilarityEngine(
            ollama_client=self.ollama,
            embedding_model=config.embeddings.model,
            dimension=config.embeddings.dimension,
        )
        self.consensus_detector = ConsensusDetector(
            threshold=config.discussion.consensus_threshold,
            method=config.discussion.consensus_method,
        )
        self.session_manager = SessionManager(config.storage.sessions_dir)
        self.progress_callback = progress_callback
        self.state = DiscussionState(
            current_round=1,
            current_model_index=0,
            model_order=[m.name for m in config.models],
        )

    async def _rotate_model_order(self, round_num: int) -> list[str]:
        model_names = [m.name for m in self.config.models]
        if self.config.discussion.rotation_order == "sequential":
            rotation = (round_num - 1) % len(model_names)
            return model_names[rotation:] + model_names[:rotation]
        elif self.config.discussion.rotation_order == "random":
            shuffled = model_names.copy()
            random.shuffle(shuffled)
            return shuffled
        else:  # "fixed" - same order every round
            return model_names

    async def _build_context(
        self,
        model_position: int,
        total_models: int,
        round_num: int,
    ) -> str:
        context_mode = self.config.context.mode

        # Helper to get human responses for a range of rounds
        def get_human_responses_for_rounds(start_round: int, end_round: int) -> list:
            human_responses = []
            for r in range(start_round, end_round + 1):
                round_humans = self.session.get_round_human_responses(r)
                human_responses.extend(round_humans)
            return human_responses

        if context_mode == "full":
            model_responses = self.session.responses
            # Get all human responses
            max_round = model_responses[-1].round if model_responses else 0
            human_responses = get_human_responses_for_rounds(1, max_round)
            all_responses = model_responses + human_responses
            if all_responses:
                context_text = "\n\n".join(
                    f"### {r.model} (Round {r.round}):\n{r.content}"
                    for r in all_responses
                )
                return f"{self.session.prompt}\n\n=== DISCUSSION HISTORY ===\n{context_text}"
            return ParticipantPrompt.initial(self.session.prompt)

        elif context_mode == "summary_only":
            latest_attributed = self.session.get_latest_attributed_summary()
            if latest_attributed and round_num > 1:
                return ParticipantPrompt.with_attributed_summary(
                    self.session.prompt,
                    latest_attributed,
                    model_position,
                    total_models,
                    round_num,
                )
            return ParticipantPrompt.initial(self.session.prompt)

        elif context_mode == "summary_plus_last_n":
            n_rounds = self.config.context.last_n_responses
            # Get last N rounds of responses, not N individual responses
            if n_rounds > 0 and self.session.responses:
                current_round = self.session.responses[-1].round
                start_round = max(1, current_round - n_rounds + 1)
                recent_model_responses = [
                    r for r in self.session.responses if r.round >= start_round
                ]
                recent_human_responses = get_human_responses_for_rounds(
                    start_round, current_round
                )
                recent_responses = recent_model_responses + recent_human_responses
            else:
                recent_responses = []
            latest_attributed = self.session.get_latest_attributed_summary()

            context_parts = []
            if latest_attributed and round_num > 1:
                individual_text = "\n\n".join(
                    f"### {model}\n" + "\n".join(f"- {p}" for p in points)
                    for model, points in latest_attributed.individual_summaries.items()
                )
                context_parts.append(
                    f"=== DISCUSSION ANALYSIS ===\n{individual_text}\n\n"
                    f"AGREEMENT: {latest_attributed.agreement_analysis}\n\n"
                    f"CONSENSUS: {latest_attributed.consensus_assessment}"
                )
            if recent_responses:
                context_text = "\n\n".join(
                    f"### {r.model}:\n{r.content}" for r in recent_responses
                )
                context_parts.append(f"=== RECENT RESPONSES ===\n{context_text}")

            if context_parts and round_num > 1:
                context = "\n\n".join(context_parts)
                return f"{self.session.prompt}\n\n{context}\n\nYou are model {model_position} of {total_models} in round {round_num}. Consider the above and provide your updated response."

            return ParticipantPrompt.initial(self.session.prompt)

        return ParticipantPrompt.initial(self.session.prompt)

    async def _handle_human_input(
        self,
        context: str,
        round_num: int,
        position: int,
    ) -> str:
        if self.human_input_callback:
            return await self.human_input_callback(context, round_num, position)

        if self.config.human_participant.enabled:
            prompt_text = self.config.human_participant.prompt.replace(
                "{prompt}", self.session.prompt
            )

            # The 'context' passed in is already the result of _build_context
            context_display = context
            if not context_display:
                context_display = "No context available."
            # Dynamic truncation based on terminal size, with sensible minimum/maximum
            term_width = (
                shutil.get_terminal_size().columns
                if shutil.get_terminal_size().columns > 0
                else 80
            )
            max_chars = max(
                500, min(term_width * 30, 5000)
            )  # 30 chars per line, between 500-5000
            if len(context_display) > max_chars:
                context_display = context_display[:max_chars] + "... [Truncated]"

            paragraphs = []

            print("\n" + "=" * 60)
            print(f"ROUND {round_num}: Your turn to respond")
            print("=" * 60)
            print(f"Prompt: {prompt_text}")
            print(f"\n{context_display}")
            print("-" * 60)
            print("\nType your response. Press Enter after each paragraph.")
            print(
                "Type text and Enter to add another paragraph, or press Enter on empty line to submit."
            )
            print("Type 's' at any time to skip. (Ctrl+C stops the program.)")

            while True:
                user_input = await asyncio.to_thread(sys.stdin.readline)

                if not user_input:
                    break

                # Skip (before empty check so 's' works even on first line)
                if user_input.strip().lower() == "s":
                    print(f"\n[Round {round_num}] Human skipped")
                    return ""

                # Empty line = submit (only if we have at least one paragraph)
                if not user_input.strip() and paragraphs:
                    break

                # Any non-empty line = next paragraph
                if user_input.strip():
                    paragraphs.append(user_input.rstrip("\n"))
                    print("\n" + "-" * 40)
                    print(f"Paragraph {len(paragraphs)} confirmed.")
                    print(
                        "Type text and Enter for another paragraph, or press Enter to submit.\n"
                    )

                # Keep reading

            # Reassemble with blank line between paragraphs
            response = "\n\n".join(paragraphs)

            if not response.strip():
                print(f"[Round {round_num}] Empty input, skipping")
                return ""

            print(f"\n[Round {round_num}] Submitted ({len(response)} chars)")
            return response

        return ""

    async def _generate_summary(
        self,
        round_num: int,
        similarity_matrix: np.ndarray | None = None,
        model_names: list[str] | None = None,
    ) -> str:
        round_responses = self.session.get_round_responses(round_num)
        human_responses = self.session.get_round_human_responses(round_num)

        all_responses = round_responses + human_responses
        if not all_responses:
            return ""

        responses_data = [
            {"model": r.model, "content": r.content, "round": r.round}
            for r in all_responses
        ]

        threshold = self.config.discussion.consensus_threshold

        if similarity_matrix is not None and model_names:
            prompt = ModeratorPrompt.template_with_similarity_matrix(
                responses_data, round_num, similarity_matrix, model_names, threshold
            )
        else:
            prompt = ModeratorPrompt.template(responses_data, round_num)

        # Check if web search is enabled for the moderator
        use_tools = self.config.tools.web_search.enabled

        # Get list of tool names for the system prompt
        tool_names = ["web_search"] if use_tools else None

        if use_tools:
            # Use chat method with tools
            tools_config = self.config.tools.model_dump()
            tools = get_available_tools(tools_config)
            tool_executor = create_tool_executor(tools_config)

            messages = [
                {
                    "role": "system",
                    "content": ModeratorPrompt.system(threshold, tools=tool_names),
                },
                {"role": "user", "content": prompt},
            ]

            response = await self.ollama.chat(
                model=self.config.moderator.name,
                messages=messages,
                tools=tools if tools else None,
                tool_executor=tool_executor,
                temperature=self.config.moderator.temperature,
                max_tokens=self.config.moderator.max_tokens,
                num_ctx=self.config.moderator.num_ctx,
            )

            full_text = response.message

            # Log if tools were used
            if response.tool_calls:
                print(
                    f"[Round {round_num}] Moderator used {len(response.tool_calls)} tool call(s)"
                )
                for tc in response.tool_calls:
                    print(f"  - {tc.name}: {tc.arguments.get('query', 'N/A')}")
        else:
            # Use generate method (no tools)
            response = await self.ollama.generate(
                model=self.config.moderator.name,
                prompt=prompt,
                system=ModeratorPrompt.system(threshold),
                temperature=self.config.moderator.temperature,
                max_tokens=self.config.moderator.max_tokens,
                num_ctx=self.config.moderator.num_ctx,
            )
            full_text = response.response

        parsed = SummaryParser.parse(full_text, all_responses)

        self.session.add_attributed_summary(
            round_num=round_num,
            individual_summaries=parsed["individual_summaries"],
            agreement_analysis=parsed["agreement_analysis"],
            consensus_assessment=parsed["consensus_assessment"],
            confidence=parsed["confidence"],
            full_text=full_text,
        )

        # Additional validation: check for contradictions between consensus_assessment and full_text
        # The full_text contains the moderator's original text which may have the definitive verdict
        if parsed["consensus_assessment"] == "REACHED":
            full_text_upper = full_text.upper()
            if re.search(
                r"consensus\s*:\s*not\s*reached", full_text_upper
            ) or re.search(r"consensus\s+not\s+reached", full_text_upper):
                print(
                    "[Warning] Parser set consensus_assessment to REACHED but full_text contains NOT REACHED"
                )
                print("  - Updating consensus_assessment to NOT REACHED")
                # Update the session with the corrected value
                attributed = self.session.get_attributed_summary(round_num)
                if attributed:
                    attributed.consensus_assessment = "NOT REACHED"

        return full_text

    async def _generate_final_review(self) -> str:
        """Generate comprehensive final review after consensus is reached."""
        print("Generating final review...")

        all_responses = self.session.responses
        all_summaries = self.session.attributed_summaries

        if not all_responses:
            return "(No responses to analyze)"

        system_prompt, user_prompt = ModeratorPrompt.final_review(
            self.session.prompt,
            all_responses,
            all_summaries,
        )

        # Check if web search is enabled for the moderator
        use_tools = self.config.tools.web_search.enabled

        # Add tool instructions to system prompt if web search is enabled
        if use_tools:
            tool_instructions = """

TOOLS AVAILABLE:
You have access to a web search tool that can search Wikipedia for factual information.
- Use web_search when you need to verify facts, statistics, or dates
- Search Wikipedia to confirm information before including in your review
"""
            system_prompt = system_prompt + tool_instructions

        if use_tools:
            # Use chat method with tools
            tools_config = self.config.tools.model_dump()
            tools = get_available_tools(tools_config)
            tool_executor = create_tool_executor(tools_config)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.ollama.chat(
                model=self.config.moderator.name,
                messages=messages,
                tools=tools if tools else None,
                tool_executor=tool_executor,
                temperature=self.config.moderator.temperature,
                max_tokens=self.config.moderator.max_tokens,
                num_ctx=self.config.moderator.num_ctx,
            )

            return response.message
        else:
            # Use generate method (no tools)
            response = await self.ollama.generate(
                model=self.config.moderator.name,
                prompt=user_prompt,
                system=system_prompt,
                temperature=self.config.moderator.temperature,
                max_tokens=self.config.moderator.max_tokens,
                num_ctx=self.config.moderator.num_ctx,
            )
            return response.response

    @staticmethod
    def _parse_json_block(text: str) -> dict | None:
        """Deprecated: use SummaryParser._parse_json_block instead."""
        return SummaryParser._parse_json_block(text)

    @staticmethod
    def _parse_markdown_summary(text: str, round_responses: "list[Response]") -> dict:
        """Deprecated: use SummaryParser._parse_markdown instead."""
        return SummaryParser._parse_markdown(text, round_responses)

    def _parse_attributed_summary(
        self, text: str, round_responses: "list[Response]"
    ) -> dict:
        """Deprecated: use SummaryParser.parse instead."""
        return SummaryParser.parse(text, round_responses)

    def _check_main_point_consensus(self, attributed) -> str:
        if not attributed:
            return _ConsensusVerdict.NOT_REACHED

        if attributed.consensus_assessment == "REACHED":
            return _ConsensusVerdict.REACHED

        if attributed.consensus_assessment == "NOT REACHED":
            agreement_analysis = attributed.agreement_analysis
            if (
                not agreement_analysis
                or agreement_analysis == "(Analysis not provided)"
            ):
                return _ConsensusVerdict.NOT_REACHED

            agreement_lower = agreement_analysis.lower()

            # Only flag INCONSISTENT when the analysis explicitly says ALL participants
            # agree on the MAIN ANSWER (not just the premise, not just 2 of 3)
            if re.search(
                r"(all.*participants?|all.*clusters?).*(agree|agreement|consensus).*main.*(answer|point|question)",
                agreement_lower,
            ):
                return _ConsensusVerdict.INCONSISTENT

            # Also catch: "main answer.*agreed" with explicit universal quantification
            if re.search(
                r"main.*(answer|point|question).*agreed.*(all|unanimously)",
                agreement_lower,
            ):
                return _ConsensusVerdict.INCONSISTENT

            return _ConsensusVerdict.NOT_REACHED

        return _ConsensusVerdict.NOT_REACHED

    def _calculate_agreement_percentage(
        self, sim_matrix: np.ndarray, threshold: float
    ) -> float:
        if sim_matrix is None or sim_matrix.size == 0:
            return 0.0

        n = sim_matrix.shape[0]
        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            return 0.0

        above_threshold = 0
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    above_threshold += 1

        return (above_threshold / total_pairs) * 100

    async def _reprompt_for_consensus(
        self,
        round_num: int,
        attributed,
        similarity_matrix: np.ndarray,
        model_names: list[str],
    ) -> str:
        threshold = self.config.discussion.consensus_threshold
        agreement_pct = self._calculate_agreement_percentage(
            similarity_matrix, threshold
        )

        reprompt = f"""The similarity matrix shows {agreement_pct:.1f}% of pairwise similarities exceed the {threshold} threshold, indicating strong agreement among participants.

Your previous assessment was: {attributed.consensus_assessment}

Based on this quantitative evidence, do you:
1. Keep your original assessment, OR
2. Change your assessment to REACHED (if the main answer is agreed upon despite peripheral disagreements)

Respond with ONLY "KEEP" or "CHANGE" followed by the word "REACHED" or "NOT REACHED" and a brief justification."""

        response = await self.ollama.generate(
            model=self.config.moderator.name,
            prompt=reprompt,
            system="You are reconsidering a consensus assessment based on new evidence.",
            temperature=0.3,
            max_tokens=200,
            num_ctx=self.config.moderator.num_ctx,
        )

        response_text = response.response.upper()

        if "CHANGE" in response_text and "REACHED" in response_text:
            print(
                f"[Round {round_num}] Reprompt: Moderator changed to REACHED based on similarity evidence"
            )
            return "REACHED"
        else:
            print(f"[Round {round_num}] Reprompt: Moderator kept original assessment")
            return attributed.consensus_assessment

    async def _check_consensus(
        self, round_num: int, return_matrix: bool = False
    ) -> ConsensusResult:
        round_responses = self.session.get_round_responses(round_num)
        human_responses = self.session.get_round_human_responses(round_num)

        all_responses = round_responses + human_responses
        if len(all_responses) < 2:
            return ConsensusResult(
                reached=False,
                percentage=0,
                agreeing_pairs=0,
                total_pairs=0,
                method=self.consensus_detector.method,
            )

        attributed = self.session.get_attributed_summary(round_num)
        summary = self.session.get_summary(round_num)

        if return_matrix:
            texts = [r.content for r in all_responses]
            model_names = [r.model for r in all_responses]

            similarity_result = (
                await self.similarity_engine.calculate_similarity_matrix(
                    texts, model_names
                )
            )

            return ConsensusResult(
                reached=False,
                percentage=0,
                agreeing_pairs=0,
                total_pairs=0,
                method=self.consensus_detector.method,
                details={
                    "similarity_matrix": similarity_result.matrix.tolist(),
                    "model_names": model_names,
                },
            )

        if summary and attributed:
            summary_embedding = await self.similarity_engine.get_embedding(summary)

            # NOTE: Sequential embedding calls (no Ollama batch API). If batch support is
            # added, parallelize with asyncio.gather().
            similarities_to_summary = []
            for r in all_responses:
                resp_embedding = await self.similarity_engine.get_embedding(r.content)
                sim = self.similarity_engine.cosine_similarity(
                    resp_embedding, summary_embedding
                )
                similarities_to_summary.append((r.model, sim))

            if attributed.consensus_assessment == "REACHED":
                threshold = self.config.discussion.consensus_agreement_when_reached
            else:
                threshold = self.config.discussion.consensus_agreement_when_not_reached

            agreeing = sum(1 for _, sim in similarities_to_summary if sim >= threshold)
            percentage = (agreeing / len(all_responses)) * 100

            if self.consensus_detector.method == "clustering":
                reaching = percentage > 50
            else:
                reaching = agreeing == len(all_responses)

            return ConsensusResult(
                reached=reaching,
                percentage=percentage,
                agreeing_pairs=agreeing,
                total_pairs=len(all_responses),
                method=self.consensus_detector.method,
                details={
                    "similarities_to_summary": similarities_to_summary,
                    "summary_length": len(summary),
                    "moderator_assessment": attributed.consensus_assessment,
                    "moderator_confidence": attributed.confidence,
                    "threshold_used": threshold,
                },
            )

        texts = [r.content for r in all_responses]
        model_names = [r.model for r in all_responses]

        similarity_result = await self.similarity_engine.calculate_similarity_matrix(
            texts, model_names
        )

        return self.consensus_detector.detect(similarity_result.matrix)

    async def _notify_progress(self) -> None:
        if self.progress_callback:
            await self.progress_callback(self.state)

    async def run(self) -> Session:
        if self.session.status == "completed":
            return self.session

        self.state.is_running = True

        # Restore state from session history
        start_round = self.session.completed_rounds + 1
        if self.session.responses:
            max_resp_round = max(r.round for r in self.session.responses)
            if max_resp_round > self.session.completed_rounds:
                start_round = max_resp_round

        # Determine starting model index if we are resuming a partial round
        current_model_index = 0
        # Calculate this regardless of start_round since we might be resuming internally in Round 1
        current_order = await self._rotate_model_order(start_round)
        responded_in_round = {
            r.model for r in self.session.responses if r.round == start_round
        }
        responded_in_round.update(
            {r.model for r in self.session.human_responses if r.round == start_round}
        )

        idx = 0
        while idx < len(current_order) and current_order[idx] in responded_in_round:
            idx += 1
        current_model_index = idx

        self.state.current_round = start_round
        self.state.current_model_index = current_model_index

        try:
            for round_num in range(start_round, self.config.discussion.max_rounds + 1):
                self.state.current_round = round_num
                self.state.model_order = await self._rotate_model_order(round_num)

                if not self.state.is_running:
                    break

                await self._notify_progress()

                if self.config.human_participant.enabled:
                    total_participants = len(self.state.model_order) + 1
                else:
                    total_participants = len(self.state.model_order)

                # Start from the restored model index if it's the starting round
                start_idx = (
                    self.state.current_model_index if round_num == start_round else 0
                )
                for i in range(start_idx, len(self.state.model_order)):
                    if not self.state.is_running:
                        break

                    if self.state.skip_requested:
                        print(
                            f"[Round {round_num}] Skip requested. Jumping to summary."
                        )
                        self.state.skip_requested = False
                        break

                    model_name = self.state.model_order[i]

                    context = await self._build_context(
                        model_position=i + 1,
                        total_models=total_participants,
                        round_num=round_num,
                    )

                    model_config_obj = next(
                        (m for m in self.config.models if m.name == model_name),
                        self.config.models[0],
                    )

                    print(f"[Round {round_num}] {model_name} responding...")

                    # Retry logic for empty responses
                    max_retries = 2
                    retry_count = 0
                    generated = None

                    while retry_count <= max_retries:
                        generated = await self.ollama.generate(
                            model=model_name,
                            prompt=context,
                            system=ParticipantPrompt.system(),
                            temperature=model_config_obj.temperature,
                            max_tokens=model_config_obj.max_tokens,
                            num_ctx=model_config_obj.num_ctx,
                            images=self.session.images,
                        )

                        # Check if response is empty
                        if generated.response and generated.response.strip():
                            break  # Got valid response

                        retry_count += 1
                        if retry_count <= max_retries:
                            print(
                                f"[Round {round_num}] {model_name} returned empty response, retrying ({retry_count}/{max_retries})..."
                            )
                        else:
                            print(
                                f"[Round {round_num}] {model_name} still returned empty after {max_retries} retries, proceeding with empty content"
                            )

                    response_text = generated

                    response_time_s = None
                    if (
                        hasattr(generated, "total_duration")
                        and generated.total_duration is not None
                    ):
                        response_time_s = round(
                            generated.total_duration / 1_000_000_000, 2
                        )

                    print(
                        f"[Round {round_num}] {model_name} completed ({len(response_text.response)} chars, {response_time_s}s)"
                    )

                    model_response_time_s = response_time_s

                    response = Response(
                        model=model_name,
                        content=response_text.response,
                        round=round_num,
                        timestamp=datetime.now().isoformat(),
                        position=i,
                        response_time_s=model_response_time_s,
                    )
                    self.session.add_response(
                        model=response.model,
                        content=response.content,
                        round_num=response.round,
                        position=response.position,
                        response_time_s=model_response_time_s,
                    )

                    if self.config.storage.auto_save:
                        await self.session_manager.save(self.session)

                    await self._notify_progress()

                if not self.state.is_running:
                    break

                if self.config.human_participant.enabled:
                    human_context = await self._build_context(
                        model_position=len(self.state.model_order) + 1,
                        total_models=len(self.state.model_order) + 1,
                        round_num=round_num,
                    )

                    # Check if human already responded in this round
                    human_already_responded = any(
                        r.round == round_num for r in self.session.human_responses
                    )

                    user_input = ""
                    if not human_already_responded:
                        user_input = await self._handle_human_input(
                            context=human_context,
                            round_num=round_num,
                            position=len(self.state.model_order),
                        )

                    if user_input:
                        self.session.add_human_response(
                            content=user_input,
                            round_num=round_num,
                            position=len(self.state.model_order),
                        )

                        if self.config.storage.auto_save:
                            await self.session_manager.save(self.session)

                print(f"[Round {round_num}] Moderator generating summary...")

                # Only generate summary if it doesn't exist for this round
                if self.session.get_summary(round_num) is None:
                    raw_consensus = await self._check_consensus(
                        round_num, return_matrix=True
                    )
                    sim_matrix_raw = raw_consensus.details.get("similarity_matrix")
                    sim_matrix = (
                        np.array(sim_matrix_raw) if sim_matrix_raw is not None else None
                    )
                    sim_names = raw_consensus.details.get("model_names", [])

                    if sim_matrix is not None:
                        self.session.add_similarity_matrix(
                            round_num, sim_matrix.tolist(), sim_names
                        )

                    summary = await self._generate_summary(
                        round_num, similarity_matrix=sim_matrix, model_names=sim_names
                    )
                    self.session.add_summary(round_num, summary)

                    print(
                        f"[Round {round_num}] Summary completed ({len(summary)} chars)"
                    )

                    attributed = self.session.get_attributed_summary(round_num)
                    if attributed:
                        print(
                            f"[Round {round_num}] Moderator assessment: {attributed.consensus_assessment} (Confidence: {attributed.confidence})"
                        )
                        print(f"[Round {round_num}] Individual summaries:")
                        for model, points in attributed.individual_summaries.items():
                            print(
                                f"  {model}: {points[0] if points else '(no points)'}"
                            )

                    if self.config.storage.auto_save:
                        await self.session_manager.save(self.session)
                else:
                    print(
                        f"[Round {round_num}] Summary already exists, skipping generation"
                    )
                    attributed = self.session.get_attributed_summary(round_num)
                    sim = self.session.get_similarity_matrix(round_num)

                    # Validate required data exists for consensus check
                    if not attributed:
                        print(
                            f"[Round {round_num}] Warning: Summary exists but attributed summary missing, regenerating..."
                        )
                        # Fall through to generate new summary
                        attributed = None
                        sim_matrix = None
                        sim_names = []
                    elif sim is None:
                        print(
                            f"[Round {round_num}] Warning: Summary exists but similarity matrix missing, regenerating..."
                        )
                        # Fall through to generate new similarity matrix
                        sim_matrix = None
                        sim_names = []
                    else:
                        sim_matrix = np.array(sim["matrix"])
                        sim_names = sim["model_names"]

                if self.config.discussion.mode == "moderator_decides":
                    if self.config.discussion.strictness == "main_point":
                        verdict = self._check_main_point_consensus(attributed)
                        consensus_reached = False  # default, updated below if needed

                        if verdict is _ConsensusVerdict.REACHED:
                            consensus_reached = True
                        elif verdict is _ConsensusVerdict.INCONSISTENT:
                            pass  # fall through to reprompt path
                        # else: verdict is NOT_REACHED → stays False

                        if (
                            verdict is not _ConsensusVerdict.REACHED
                            and attributed
                            and sim_matrix is not None
                        ):
                            threshold = self.config.discussion.consensus_threshold
                            agreement_pct = self._calculate_agreement_percentage(
                                sim_matrix, threshold
                            )

                            # Option 3: Only reprompt on specific conditions (contradictions), not just similarity
                            # Check for clear contradictions in the moderator's reasoning
                            full_text = (
                                attributed.full_text
                                if hasattr(attributed, "full_text")
                                and attributed.full_text
                                else ""
                            )
                            agreement_text = (
                                attributed.agreement_analysis
                                if attributed.agreement_analysis
                                else ""
                            )
                            combined_text = (full_text + " " + agreement_text).upper()

                            # Only reprompt if there's a CLEAR CONTRADICTION in the moderator's reasoning
                            has_contradiction = (
                                re.search(
                                    r"(all|everyone|participants)\s+(agree|agree\w*)",
                                    agreement_text,
                                    re.I,
                                )
                                and re.search(
                                    r"consensus\s*:\s*not\s*reached", combined_text
                                )
                            ) or (
                                re.search(r"no\s+disagreement", agreement_text, re.I)
                                and re.search(
                                    r"consensus\s*:\s*not\s*reached", combined_text
                                )
                            )

                            if has_contradiction:
                                print(
                                    f"[Round {round_num}] Contradiction detected in moderator reasoning. Reprompting for clarity..."
                                )
                                revised = await self._reprompt_for_consensus(
                                    round_num, attributed, sim_matrix, sim_names
                                )
                                # Option 2: Reprompt is advisory only - moderator can consider similarity but decides
                                # Only update if moderator explicitly changes their decision
                                if (
                                    revised
                                    and revised != attributed.consensus_assessment
                                ):
                                    print(
                                        f"  - Similarity agreement: {agreement_pct:.1f}%"
                                    )
                                    print(f"  - Moderator revised to: {revised}")
                                    attributed.consensus_assessment = revised
                                    consensus_reached = revised == "REACHED"
                            else:
                                # No contradiction - keep moderator's original decision
                                print(
                                    f"[Round {round_num}] Moderator assessment stands (no contradiction detected)"
                                )

                        # Build ConsensusResult for TUI state (fix #1.1)
                        _agreement_pct = 0.0
                        if sim_matrix is not None:
                            _agreement_pct = self._calculate_agreement_percentage(
                                sim_matrix, self.config.discussion.consensus_threshold
                            )
                        _n = len(sim_names) if sim_names else 0
                        _total_pairs = _n * (_n - 1) // 2 if _n > 1 else 0
                        self.state.consensus_result = ConsensusResult(
                            reached=consensus_reached,
                            percentage=_agreement_pct,
                            agreeing_pairs=int(_agreement_pct / 100 * _total_pairs)
                            if _total_pairs > 0
                            else 0,
                            total_pairs=_total_pairs,
                            method=self.config.discussion.consensus_method,
                        )
                    else:
                        consensus_reached = (
                            attributed.consensus_assessment == "REACHED"
                            if attributed
                            else False
                        )

                        # Build ConsensusResult for TUI state (fix #1.1)
                        if sim_matrix is not None:
                            _agreement_pct = self._calculate_agreement_percentage(
                                sim_matrix, self.config.discussion.consensus_threshold
                            )
                        else:
                            _agreement_pct = 100.0 if consensus_reached else 0.0
                        _n = len(sim_names) if sim_names else 0
                        _total_pairs = _n * (_n - 1) // 2 if _n > 1 else 0
                        self.state.consensus_result = ConsensusResult(
                            reached=consensus_reached,
                            percentage=_agreement_pct,
                            agreeing_pairs=int(_agreement_pct / 100 * _total_pairs)
                            if _total_pairs > 0
                            else 0,
                            total_pairs=_total_pairs,
                            method=self.config.discussion.consensus_method,
                        )
                else:
                    consensus_result = await self._check_consensus(round_num)
                    self.state.consensus_result = consensus_result
                    consensus_reached = consensus_result.reached

                print(
                    f"[Round {round_num}] Consensus: {'REACHED' if consensus_reached else 'NOT REACHED'}"
                )

                if consensus_reached:
                    if self.config.discussion.final_review_enabled:
                        try:
                            final_review = await self._generate_final_review()
                            self.session.add_final_review(final_review)
                        except Exception as e:
                            print(f"Warning: Final review generation failed: {e}")
                            print("Continuing without final review.")

                    self.session.mark_completed(consensus_round=round_num)
                    await self.session_manager.save(self.session)
                    break

            if self.session.status == "running":
                self.session.mark_completed()
                await self.session_manager.save(self.session)

        except Exception as e:
            print(
                f"\n[ERROR] Discussion failed during round "
                f"{self.state.current_round}: {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc()
            self.session.mark_stopped()
            await self.session_manager.save(self.session)
            raise

        finally:
            self.state.is_running = False
            await self._notify_progress()

        return self.session

    async def pause(self) -> None:
        self.state.is_paused = True
        await self._notify_progress()

    async def resume(self) -> None:
        self.state.is_paused = False
        await self._notify_progress()

    async def stop(self) -> None:
        self.state.is_running = False
        self.state.is_paused = False
        await self._notify_progress()

    async def cleanup(self) -> None:
        await self.ollama.close()
