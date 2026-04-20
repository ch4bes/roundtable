import asyncio
from datetime import datetime
from typing import Callable, Awaitable
from dataclasses import dataclass
import numpy as np
from .config import Config
from .ollama_client import OllamaClient
from .similarity import SimilarityEngine
from .consensus import ConsensusDetector, ConsensusResult
from .input_reader import AsyncInputReader, InputBuffer, get_input_buffer
from prompts.system_prompts import ModeratorPrompt, ParticipantPrompt
from storage.session import Session, SessionManager, Response


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
            return ParticipantPrompt.initial(self.session.prompt, model_position, total_models)

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
            return ParticipantPrompt.initial(self.session.prompt, model_position, total_models)

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

            return ParticipantPrompt.initial(self.session.prompt, model_position, total_models)

        return ParticipantPrompt.initial(self.session.prompt, model_position, total_models)

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
            prompt_text = self.config.human_participant.prompt.format(prompt=self.session.prompt)
            print(f"\n{'=' * 60}")
            print(f"ROUND {round_num}: Your turn to respond")
            print(f"{'=' * 60}")
            print(f"Prompt: {prompt_text}")
            print(f"\nContext from discussion:")
            print(f"{context[:500]}..." if len(context) > 500 else f"\n{context}")
            print(f"{'-' * 60}")

            if self.input_buffer is not None:
                async_reader = AsyncInputReader(prompt_text="\nYour response: ")
                async_reader.start()

                print("(Type your response and press Enter...)")
                user_input = async_reader.get_input(timeout=60.0)

                if user_input is not None:
                    print(f"[Round {round_num}] Human completed ({len(user_input)} chars)")
                    return user_input

                async_reader.stop()
                print("(No async input received, falling back to prompt)")

            human_input = input("\nYour response: ")
            return human_input

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

    def _parse_attributed_summary(self, text: str, round_responses: list) -> dict:
        individual_summaries: dict[str, list[str]] = {}
        agreement_analysis = ""
        consensus_assessment = "NOT REACHED"
        confidence = "MEDIUM"

        lines = text.split("\n")
        current_model = None

        final_consensus_section_started = False
        for line in lines:
            line = line.strip()

            if line.startswith("### "):
                current_model = line[4:].strip()
                individual_summaries[current_model] = []
            elif line.startswith("- ") and current_model:
                individual_summaries[current_model].append(line[2:])
            elif line.startswith("## Agreement") or "Agreement Analysis" in line:
                current_model = None
            elif line.startswith("## Final Consensus") or "Final Consensus" in line:
                current_model = None
                final_consensus_section_started = True
            elif line.startswith("## Consensus") or line.startswith("## Final"):
                current_model = None
                final_consensus_section_started = True
            elif final_consensus_section_started and "Consensus:" in line:
                if "NOT REACHED" in line.upper():
                    consensus_assessment = "NOT REACHED"
                elif "REACHED" in line.upper():
                    consensus_assessment = "REACHED"
            elif not final_consensus_section_started and ("Consensus Assessment:" in line or "**Consensus Assessment:**" in line):
                if "NOT REACHED" in line.upper():
                    consensus_assessment = "NOT REACHED"
                elif "REACHED" in line.upper():
                    consensus_assessment = "REACHED"
            elif "Consensus:" in line:
                if "NOT REACHED" in line.upper():
                    consensus_assessment = "NOT REACHED"
                elif "REACHED" in line.upper() and not final_consensus_section_started:
                    consensus_assessment = "REACHED"
            elif "Confidence:" in line or "CONFIDENCE:" in line:
                if "HIGH" in line.upper():
                    confidence = "HIGH"
                elif "LOW" in line.upper():
                    confidence = "LOW"
                else:
                    confidence = "MEDIUM"
            elif current_model is None and line and not line.startswith("#"):
                if not agreement_analysis:
                    agreement_analysis = line
                else:
                    agreement_analysis += "\n" + line

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

    def _check_main_point_consensus(self, attributed) -> bool:
        if not attributed:
            return False

        if attributed.consensus_assessment == "REACHED":
            return True

        if attributed.consensus_assessment == "NOT REACHED":
            agreement_analysis = attributed.agreement_analysis
            if not agreement_analysis or agreement_analysis == "(Analysis not provided)":
                return False

            agreement_lower = agreement_analysis.lower()

            full_agreement_patterns = [
                "areas of full agreement",
                "overwhelming consensus",
                "unanimously agree",
                "all participants agree",
                "strong consensus",
                "no disagreement",
                "all clusters agree",
            ]

            has_full_agreement = any(pattern in agreement_lower for pattern in full_agreement_patterns)

            if has_full_agreement:
                return True

            main_point_patterns = [
                "main answer",
                "peripheral",
            ]

            has_main_point_mention = any(pattern in agreement_lower for pattern in main_point_patterns)

            if has_main_point_mention and has_full_agreement:
                return True

            if "consensus" in agreement_lower and "not reached" not in agreement_lower:
                return True

        return False

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

            agreement_lower = agreement_analysis.lower()

            full_agreement_patterns = [
                "areas of full agreement",
                "overwhelming consensus",
                "unanimously agree",
                "all participants agree",
                "strong consensus",
                "no disagreement",
                "all clusters agree",
            ]

            has_full_agreement = any(pattern in agreement_lower for pattern in full_agreement_patterns)

            if has_full_agreement:
                return True

            main_point_patterns = [
                "main answer",
                "peripheral",
            ]

            has_main_point_mention = any(pattern in agreement_lower for pattern in main_point_patterns)

            if has_main_point_mention and has_full_agreement:
                return True

            if "consensus" in agreement_lower and "not reached" not in agreement_lower:
                return True

        return False

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

            similarities_to_summary = []
            for r in all_responses:
                resp_embedding = await self.similarity_engine.get_embedding(r.content)
                sim = self.similarity_engine.cosine_similarity(resp_embedding, summary_embedding)
                similarities_to_summary.append((r.model, sim))

            if attributed.consensus_assessment == "REACHED":
                threshold = 0.50
            else:
                threshold = 0.75

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

                # Start human input capture early (during model responses)
                human_reader = None
                if self.config.human_participant.enabled:
                    human_reader = AsyncInputReader(
                        prompt_text="\nYour response (type while models respond): "
                    )
                    human_reader.start()
                    print(f"[Round {round_num}] Human can type while models respond...")

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

                    if human_reader:
                        user_input = human_reader.get_input(timeout=90.0)
                        if user_input is None or user_input.strip() == "":
                            print("(No input received during model response)")
                            print(f"\n{'=' * 60}")
                            print(f"ROUND {round_num}: Your turn to respond")
                            print(f"{'=' * 60}")
                            prompt_text = self.config.human_participant.prompt.format(
                                prompt=self.session.prompt
                            )
                            print(f"Prompt: {prompt_text}")
                            print(f"\nContext from discussion:")
                            print(
                                f"{human_context[:500]}..."
                                if len(human_context) > 500
                                else f"\n{human_context}"
                            )
                            print(f"{'-' * 60}")
                            user_input = input("\nYour response: ")
                        human_reader.stop()
                    else:
                        user_input = await self._handle_human_input(
                            context=human_context,
                            round_num=round_num,
                            position=len(self.state.model_order),
                        )

                    print(f"[Round {round_num}] Human completed ({len(user_input)} chars)")

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
                        consensus_reached = self._check_main_point_consensus(attributed)

                        if not consensus_reached and attributed and sim_matrix is not None:
                            threshold = self.config.discussion.consensus_threshold
                            agreement_pct = self._calculate_agreement_percentage(sim_matrix, threshold)

                            if agreement_pct >= 70:
                                print(f"[Round {round_num}] High agreement ({agreement_pct:.1f}%) but moderator said NOT REACHED. Reprompting...")
                                revised = await self._reprompt_for_consensus(
                                    round_num, attributed, sim_matrix, sim_names
                                )
                                if revised == "REACHED":
                                    consensus_reached = True
                                    attributed.consensus_assessment = "REACHED"
                    else:
                        consensus_reached = (
                            attributed.consensus_assessment == "REACHED" if attributed else False
                        )
                else:
                    consensus_result = await self._check_consensus(round_num)
                    self.state.consensus_result = consensus_result
                    consensus_reached = consensus_result.reached

                print(
                    f"[Round {round_num}] Consensus: {'REACHED' if consensus_reached else 'NOT REACHED'}"
                )

                if consensus_reached:
                    self.session.mark_completed(consensus_round=round_num)
                    await self.session_manager.save(self.session)
                    break

            if self.session.status == "running":
                self.session.mark_completed()
                await self.session_manager.save(self.session)

        except Exception as e:
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
