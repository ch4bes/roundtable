import asyncio
import json
import re
import sys
from datetime import datetime
from typing import Callable, Awaitable
from dataclasses import dataclass
import numpy as np
from .config import Config
from .ollama_client import OllamaClient
from .similarity import SimilarityEngine
from .consensus import ConsensusDetector, ConsensusResult
from .input_reader import InputBuffer, get_input_buffer
from prompts.system_prompts import ModeratorPrompt, ParticipantPrompt
from storage.session import Session, SessionManager, Response


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


class DiscussionOrchestrator:
    def __init__(
        self,
        config: Config,
        session: Session,
        progress_callback: Callable[[DiscussionState], Awaitable[None]] | None = None,
        human_input_callback: Callable[[str, int, int], Awaitable[str]] | None = None,
        input_buffer: InputBuffer | None = None,
    ):
        self.config = config
        self.session = session
        self.human_input_callback = human_input_callback
        self.input_buffer = input_buffer
        self.ollama = OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
        )
        self.similarity_engine = SimilarityEngine(
            ollama_client=self.ollama,
            embedding_model=config.embeddings.model,
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
        return model_names

    async def _build_context(
        self,
        model_position: int,
        total_models: int,
        round_num: int,
    ) -> str:
        context_mode = self.config.context.mode

        if context_mode == "full":
            all_responses = self.session.responses
            if all_responses:
                context_text = "\n\n".join(
                    f"### {r.model} (Round {r.round}):\n{r.content}" for r in all_responses
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
            n = self.config.context.last_n_responses
            recent_responses = self.session.responses[-n:] if n > 0 else []
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
                context_text = "\n\n".join(f"### {r.model}:\n{r.content}" for r in recent_responses)
                context_parts.append(f"=== RECENT RESPONSES ===\n{context_text}")

            if context_parts and round_num > 1:
                context = "\n\n".join(context_parts)
                return f"{self.session.prompt}\n\n{context}\n\nYou are model {model_position} of {total_models} in round {round_num}. Consider the above and provide your updated response."

            return ParticipantPrompt.initial(self.session.prompt)

        return ParticipantPrompt.initial(self.session.prompt)

    async def _generate_response(
        self,
        model_name: str,
        context: str,
        model_config,
    ) -> str:
        model_config_obj = next(
            (m for m in self.config.models if m.name == model_name),
            self.config.models[0],
        )

        response = await self.ollama.generate(
            model=model_name,
            prompt=context,
            system=ParticipantPrompt.system(),
            temperature=model_config_obj.temperature,
            max_tokens=model_config_obj.max_tokens,
            num_ctx=model_config_obj.num_ctx,
        )
        return response.response

    async def _handle_human_input(
        self,
        context: str,
        round_num: int,
        position: int,
    ) -> str:
        if self.human_input_callback:
            return await self.human_input_callback(context, round_num, position)

        if self.config.human_participant.enabled:
            prompt_text = self.config.human_participant.prompt.replace("{prompt}", self.session.prompt)
            
            latest_summary = self.session.get_latest_attributed_summary()
            if latest_summary and round_num > 1:
                context_display = f"=== LATEST SUMMARY (Round {latest_summary.round}) ===\n"
                context_display += f"AGREEMENT: {latest_summary.agreement_analysis}\n\n"
                context_display += f"CONSENSUS: {latest_summary.consensus_assessment}\n\n"
                context_display += "Individual points:\n"
                for model, points in latest_summary.individual_summaries.items():
                    context_display += f"  {model}: {points[0] if points else '(no points)'}\n"
            else:
                context_display = context[:500] + "..." if len(context) > 500 else context
            
            paragraphs = []
            
            print("\n" + "=" * 60)
            print(f"ROUND {round_num}: Your turn to respond")
            print("=" * 60)
            print(f"Prompt: {prompt_text}")
            print(f"\n{context_display}")
            print("-" * 60)
            print("\nType your response. Press Enter after each paragraph.")
            print("Type text and Enter to add another paragraph, or press Enter on empty line to submit.")
            print("Type 's' at any time to skip. (Ctrl+C stops the program.)")
            
            while True:
                user_input = await asyncio.to_thread(sys.stdin.readline)
                
                # Skip (before empty check so 's' works even on first line)
                if user_input.strip().lower() == 's':
                    print(f"\n[Round {round_num}] Human skipped")
                    return ""
                
                # Empty line = submit (only if we have at least one paragraph)
                if not user_input.strip() and paragraphs:
                    break
                
                # Any non-empty line = next paragraph
                if user_input.strip():
                    paragraphs.append(user_input.rstrip('\n'))
                    print("\n" + "-" * 40)
                    print(f"Paragraph {len(paragraphs)} confirmed.")
                    print("Type text and Enter for another paragraph, or press Enter to submit.\n")
                
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
        self, round_num: int, similarity_matrix=None, model_names: list[str] = None
    ) -> str:
        round_responses = self.session.get_round_responses(round_num)
        human_responses = self.session.get_round_human_responses(round_num)

        all_responses = round_responses + human_responses
        if not all_responses:
            return ""

        responses_data = [
            {"model": r.model, "content": r.content, "round": r.round} for r in all_responses
        ]

        threshold = self.config.discussion.consensus_threshold

        if similarity_matrix is not None and model_names:
            prompt = ModeratorPrompt.template_with_similarity_matrix(
                responses_data, round_num, similarity_matrix, model_names, threshold
            )
        else:
            prompt = ModeratorPrompt.template(responses_data, round_num)

        response = await self.ollama.generate(
            model=self.config.moderator.name,
            prompt=prompt,
            system=ModeratorPrompt.system(threshold),
            temperature=self.config.moderator.temperature,
            max_tokens=self.config.moderator.max_tokens,
            num_ctx=self.config.moderator.num_ctx,
        )

        full_text = response.response

        parsed = self._parse_attributed_summary(full_text, all_responses)

        self.session.add_attributed_summary(
            round_num=round_num,
            individual_summaries=parsed["individual_summaries"],
            agreement_analysis=parsed["agreement_analysis"],
            consensus_assessment=parsed["consensus_assessment"],
            confidence=parsed["confidence"],
            full_text=full_text,
        )

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

        response = await self.ollama.generate(
            model=self.config.moderator.name,
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.config.moderator.temperature,
            max_tokens=self.config.moderator.max_tokens,
            num_ctx=self.config.moderator.num_ctx,
        )

        return response.response

    def _parse_attributed_summary(self, text: str, round_responses: list) -> dict:
        """
        Parse the moderator's attributed summary from Markdown text.
        
        Uses a two-tier approach:
        1. First tries to parse JSON blocks (preferred - most reliable)
        2. Falls back to regex-based Markdown parsing with tolerant patterns
        """
        import re
        
        # Tier 1: Try JSON block parsing first
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "individual_summaries" in parsed and "consensus_assessment" in parsed:
                    return {
                        "individual_summaries": parsed.get("individual_summaries", {}),
                        "agreement_analysis": parsed.get("agreement_analysis", "(Not provided)"),
                        "consensus_assessment": parsed.get("consensus_assessment", "NOT REACHED"),
                        "confidence": parsed.get("confidence", "MEDIUM"),
                    }
            except json.JSONDecodeError:
                pass  # Fall through to Markdown parsing
        
        # Tier 2: Regex-based Markdown parsing (tolerant of LLM variations)
        individual_summaries: dict[str, list[str]] = {}
        agreement_analysis = ""
        consensus_assessment = "NOT REACHED"
        confidence = "MEDIUM"

        lines = text.split("\n")
        current_model = None
        final_consensus_section_started = False

        for line in lines:
            stripped = line.strip()

            # HEADINGS: match 2-5 # with optional spacing
            heading_match = re.match(r'^#{2,5}\s?(.*)', stripped)
            if heading_match:
                heading_text = heading_match.group(1).strip()

                # 3+ hashes = individual model section
                if heading_match.group(0).startswith("###") and heading_text:
                    current_model = heading_text
                    individual_summaries[current_model] = []

                # Section headings (Agreement, Consensus, Final Consensus)
                elif re.search(r'agreement|consensus', heading_text, re.I):
                    current_model = None
                    if re.search(r'final\s*consensus', heading_text, re.I):
                        final_consensus_section_started = True
                else:
                    current_model = None

            # BULLETS: support - * • —
            elif current_model:
                bullet_match = re.match(r'^[-*•–—]\s+(.*)', stripped)
                if bullet_match:
                    individual_summaries[current_model].append(bullet_match.group(1).strip())
                    continue

            # CONSENSUS VERDICT: flexible spacing
            # Match "consensus:" or "Consensus Assessment:" patterns
            if re.search(r'consensus\s*:\s*', stripped, re.I) or re.search(r'consensus\s+assessment', stripped, re.I):
                if "NOT REACHED" in stripped.upper():
                    consensus_assessment = "NOT REACHED"
                elif "REACHED" in stripped.upper() and not final_consensus_section_started:
                    consensus_assessment = "REACHED"

            if final_consensus_section_started and "REACHED" in stripped.upper():
                if "NOT REACHED" in stripped.upper():
                    consensus_assessment = "NOT REACHED"
                else:
                    consensus_assessment = "REACHED"

            # CONFIDENCE: flexible spacing
            if re.search(r'confidence\s*:|CONFIDENCE:', stripped, re.I):
                if "HIGH" in stripped.upper():
                    confidence = "HIGH"
                elif "LOW" in stripped.upper():
                    confidence = "LOW"
                else:
                    confidence = "MEDIUM"

            # AGREEMENT ANALYSIS: collect text outside model sections
            elif current_model is None and stripped and not stripped.startswith("#"):
                if not agreement_analysis:
                    agreement_analysis = stripped
                else:
                    agreement_analysis += "\n" + stripped

        if not individual_summaries:
            for r in round_responses:
                individual_summaries[r.model] = ["(No specific points extracted)"]

        if not agreement_analysis:
            agreement_analysis = "(Analysis not provided)"

        return {
            "individual_summaries": individual_summaries,
            "agreement_analysis": agreement_analysis,
            "consensus_assessment": consensus_assessment,
            "confidence": confidence,
        }

    def _check_main_point_consensus(self, attributed) -> str:
        if not attributed:
            return _ConsensusVerdict.NOT_REACHED

        if attributed.consensus_assessment == "REACHED":
            return _ConsensusVerdict.REACHED

        if attributed.consensus_assessment == "NOT REACHED":
            agreement_analysis = attributed.agreement_analysis
            if not agreement_analysis or agreement_analysis == "(Analysis not provided)":
                return _ConsensusVerdict.NOT_REACHED

            agreement_lower = agreement_analysis.lower()

            # Only flag INCONSISTENT when the analysis explicitly says ALL participants
            # agree on the MAIN ANSWER (not just the premise, not just 2 of 3)
            if re.search(r'(all.*participants?|all.*clusters?).*(agree|agreement|consensus).*main.*(answer|point|question)', agreement_lower):
                return _ConsensusVerdict.INCONSISTENT

            # Also catch: "main answer.*agreed" with explicit universal quantification
            if re.search(r'main.*(answer|point|question).*agreed.*(all|unanimously)', agreement_lower):
                return _ConsensusVerdict.INCONSISTENT

            return _ConsensusVerdict.NOT_REACHED

        return _ConsensusVerdict.NOT_REACHED

    def _calculate_agreement_percentage(self, sim_matrix: np.ndarray, threshold: float) -> float:
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
        agreement_pct = self._calculate_agreement_percentage(similarity_matrix, threshold)

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
            print(f"[Round {round_num}] Reprompt: Moderator changed to REACHED based on similarity evidence")
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

            similarity_result = await self.similarity_engine.calculate_similarity_matrix(
                texts, model_names
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
                sim = self.similarity_engine.cosine_similarity(resp_embedding, summary_embedding)
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

    async def _notify_progress(self):
        if self.progress_callback:
            await self.progress_callback(self.state)

    async def run(self) -> Session:
        self.state.is_running = True

        try:
            for round_num in range(1, self.config.discussion.max_rounds + 1):
                self.state.current_round = round_num
                self.state.model_order = await self._rotate_model_order(round_num)

                if not self.state.is_running:
                    break

                await self._notify_progress()

                if self.config.human_participant.enabled:
                    total_participants = len(self.state.model_order) + 1
                else:
                    total_participants = len(self.state.model_order)

                for i, model_name in enumerate(self.state.model_order):
                    if not self.state.is_running:
                        break

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

                    generated = await self.ollama.generate(
                        model=model_name,
                        prompt=context,
                        system=ParticipantPrompt.system(),
                        temperature=model_config_obj.temperature,
                        max_tokens=model_config_obj.max_tokens,
                        num_ctx=model_config_obj.num_ctx,
                    )
                    response_text = generated

                    response_time_s = None
                    if hasattr(generated, 'total_duration') and generated.total_duration is not None:
                        response_time_s = round(generated.total_duration / 1_000_000_000, 2)

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

                raw_consensus = await self._check_consensus(round_num, return_matrix=True)
                sim_matrix_raw = raw_consensus.details.get("similarity_matrix")
                sim_matrix = np.array(sim_matrix_raw) if sim_matrix_raw is not None else None
                sim_names = raw_consensus.details.get("model_names", [])

                if sim_matrix is not None:
                    self.session.add_similarity_matrix(round_num, sim_matrix.tolist(), sim_names)

                summary = await self._generate_summary(
                    round_num, similarity_matrix=sim_matrix, model_names=sim_names
                )
                self.session.add_summary(round_num, summary)

                print(f"[Round {round_num}] Summary completed ({len(summary)} chars)")

                attributed = self.session.get_attributed_summary(round_num)
                if attributed:
                    print(
                        f"[Round {round_num}] Moderator assessment: {attributed.consensus_assessment} (Confidence: {attributed.confidence})"
                    )
                    print(f"[Round {round_num}] Individual summaries:")
                    for model, points in attributed.individual_summaries.items():
                        print(f"  {model}: {points[0] if points else '(no points)'}")

                if self.config.storage.auto_save:
                    await self.session_manager.save(self.session)

                if self.config.consensus.mode == "moderator_decides":
                    if self.config.consensus.strictness == "main_point":
                        verdict = self._check_main_point_consensus(attributed)
                        consensus_reached = False   # default, updated below if needed

                        if verdict is _ConsensusVerdict.REACHED:
                            consensus_reached = True
                        elif verdict is _ConsensusVerdict.INCONSISTENT:
                            pass   # fall through to reprompt path
                         # else: verdict is NOT_REACHED → stays False

                        if verdict is not _ConsensusVerdict.REACHED and attributed and sim_matrix is not None:
                            threshold = self.config.discussion.consensus_threshold
                            agreement_pct = self._calculate_agreement_percentage(sim_matrix, threshold)

                            if agreement_pct >= (self.config.discussion.reprompt_agreement_threshold * 100):
                                verdict_label = "NOT REACHED" if verdict is _ConsensusVerdict.NOT_REACHED else "INCONSISTENT"
                                print(f"[Round {round_num}] High agreement ({agreement_pct:.1f}%) but moderator said {verdict_label}. Reprompting...")
                                revised = await self._reprompt_for_consensus(
                                    round_num, attributed, sim_matrix, sim_names
                                 )
                                if revised == "REACHED":
                                    consensus_reached = True
                                    attributed.consensus_assessment = "REACHED"

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
                            agreeing_pairs=int(_agreement_pct / 100 * _total_pairs) if _total_pairs > 0 else 0,
                            total_pairs=_total_pairs,
                            method=self.config.consensus.method,
                           )
                    else:
                        consensus_reached = (
                            attributed.consensus_assessment == "REACHED" if attributed else False
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
                            agreeing_pairs=int(_agreement_pct / 100 * _total_pairs) if _total_pairs > 0 else 0,
                            total_pairs=_total_pairs,
                            method=self.config.consensus.method,
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

    async def pause(self):
        self.state.is_paused = True
        await self._notify_progress()

    async def resume(self):
        self.state.is_paused = False
        await self._notify_progress()

    async def stop(self):
        self.state.is_running = False
        self.state.is_paused = False
        await self._notify_progress()

    async def cleanup(self):
        await self.ollama.close()
