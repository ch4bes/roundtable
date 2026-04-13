import asyncio
from typing import Callable, Awaitable
from dataclasses import dataclass
from .config import Config
from .ollama_client import OllamaClient
from .similarity import SimilarityEngine
from .consensus import ConsensusDetector, ConsensusResult
from prompts.system_prompts import ModeratorPrompt, ParticipantPrompt
from storage.session import Session, SessionManager


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
    ):
        self.config = config
        self.session = session
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
            latest_summary = self.session.get_latest_summary()
            if latest_summary and round_num > 1:
                return ParticipantPrompt.with_summary(
                    self.session.prompt,
                    latest_summary,
                    model_position,
                    total_models,
                    round_num,
                )
            return ParticipantPrompt.initial(self.session.prompt, model_position, total_models)

        elif context_mode == "summary_plus_last_n":
            n = self.config.context.last_n_responses
            recent_responses = self.session.responses[-n:] if n > 0 else []
            latest_summary = self.session.get_latest_summary()

            context_parts = []
            if latest_summary and round_num > 1:
                context_parts.append(f"=== DISCUSSION SUMMARY ===\n{latest_summary}")
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

    async def _generate_summary(self, round_num: int) -> str:
        round_responses = self.session.get_round_responses(round_num)
        if not round_responses:
            return ""

        responses_data = [
            {"model": r.model, "content": r.content, "round": r.round} for r in round_responses
        ]

        prompt = ModeratorPrompt.template(responses_data, round_num)

        response = await self.ollama.generate(
            model=self.config.moderator.name,
            prompt=prompt,
            system=ModeratorPrompt.system(),
            temperature=self.config.moderator.temperature,
            max_tokens=self.config.moderator.max_tokens,
            num_ctx=self.config.moderator.num_ctx,
        )
        return response.response

    async def _check_consensus(self, round_num: int) -> ConsensusResult:
        round_responses = self.session.get_round_responses(round_num)
        if len(round_responses) < 2:
            return ConsensusResult(
                reached=False,
                percentage=0,
                agreeing_pairs=0,
                total_pairs=0,
                method=self.consensus_detector.method,
            )

        summary = self.session.get_summary(round_num)

        if summary:
            texts = [r.content for r in round_responses]
            model_names = [r.model for r in round_responses]

            similarity_result = await self.similarity_engine.calculate_similarity_matrix(
                texts, model_names
            )

            summary_embedding = await self.similarity_engine.get_embedding(summary)

            similarities_to_summary = []
            for r in round_responses:
                resp_embedding = await self.similarity_engine.get_embedding(r.content)
                sim = self.similarity_engine.cosine_similarity(resp_embedding, summary_embedding)
                similarities_to_summary.append((r.model, sim))

            agreeing = sum(
                1 for _, sim in similarities_to_summary if sim >= self.consensus_detector.threshold
            )
            percentage = (agreeing / len(round_responses)) * 100

            if self.consensus_detector.method == "clustering":
                reaching = percentage > 50
            else:
                reaching = agreeing == len(round_responses)

            return ConsensusResult(
                reached=reaching,
                percentage=percentage,
                agreeing_pairs=agreeing,
                total_pairs=len(round_responses),
                method=self.consensus_detector.method,
                details={
                    "similarities_to_summary": similarities_to_summary,
                    "summary_length": len(summary),
                },
            )

        texts = [r.content for r in round_responses]
        model_names = [r.model for r in round_responses]

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

                for model_idx, model_name in enumerate(self.state.model_order):
                    if self.state.is_paused:
                        while self.state.is_paused:
                            await asyncio.sleep(0.1)

                    if not self.state.is_running:
                        self.session.mark_stopped()
                        break

                    self.state.current_model_index = model_idx
                    await self._notify_progress()

                    context = await self._build_context(
                        model_position=model_idx + 1,
                        total_models=len(self.state.model_order),
                        round_num=round_num,
                    )

                    print(f"[Round {round_num}] {model_name} responding...")

                    response_text = await self._generate_response(
                        model_name=model_name,
                        context=context,
                        model_config=self.config.models[0],
                    )

                    print(
                        f"[Round {round_num}] {model_name} completed ({len(response_text)} chars)"
                    )

                    self.session.add_response(
                        model=model_name,
                        content=response_text,
                        round_num=round_num,
                        position=model_idx,
                    )

                    if self.config.storage.auto_save:
                        await self.session_manager.save(self.session)

                    await self._notify_progress()

                if not self.state.is_running:
                    break

                print(f"[Round {round_num}] Moderator generating summary...")

                summary = await self._generate_summary(round_num)
                self.session.add_summary(round_num, summary)

                print(f"[Round {round_num}] Summary completed ({len(summary)} chars)")
                print(f"[Round {round_num}] Summary:")
                for line in summary.strip().split("\n"):
                    print(f"  {line.strip()}")

                if self.config.storage.auto_save:
                    await self.session_manager.save(self.session)

                consensus_result = await self._check_consensus(round_num)
                self.state.consensus_result = consensus_result
                await self._notify_progress()

                print(
                    f"[Round {round_num}] Consensus: {consensus_result.percentage:.0f}% ({consensus_result.method}, cluster_size={consensus_result.agreeing_pairs})"
                )

                if consensus_result.reached:
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
