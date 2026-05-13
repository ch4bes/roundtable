import pytest
from unittest.mock import AsyncMock, MagicMock
from core.config import Config, ModelConfig, DiscussionConfig, ContextConfig
from core.discussion import DiscussionOrchestrator
from storage.session import Session

@pytest.mark.asyncio
async def test_resume_from_completed_round():
    """Verify that resuming a session starts at the next round if previous rounds are completed."""
    config = Config(
        models=[ModelConfig(name="m1"), ModelConfig(name="m2")],
        discussion=DiscussionConfig(max_rounds=3),
        context=ContextConfig(mode="summary_only"),
    )
    session = Session(prompt="test", config=config.model_dump())
    
    # Simulate Round 1 being fully completed
    session.add_response("m1", "resp1", 1, 0)
    session.add_response("m2", "resp2", 1, 1)
    session.add_summary(1, "Summary 1")
    # Add attributed summary so _build_context uses the summary prompt
    session.add_attributed_summary(
        round_num=1,
        individual_summaries={"m1": ["p1"], "m2": ["p2"]},
        agreement_analysis="Agreed",
        consensus_assessment="NOT REACHED",
        confidence="HIGH",
        full_text="Full text"
    )
    session.completed_rounds = 1
    
    orchestrator = DiscussionOrchestrator(config=config, session=session)
    orchestrator.ollama.generate = AsyncMock(return_value=MagicMock(response="new resp"))
    orchestrator.session_manager.save = AsyncMock()
    
    await orchestrator.run()
    
    # Verify that the first request in the resumed run is for Round 2
    # The first call to generate should be for a model in round 2
    first_call_kwargs = orchestrator.ollama.generate.call_args_list[0].kwargs
    # The prompt context should indicate round 2
    assert "round 2" in first_call_kwargs.get("prompt", "").lower() or "Round 2" in first_call_kwargs.get("prompt", "")

@pytest.mark.asyncio
async def test_resume_from_partial_round():
    """Verify that resuming from a partial round starts with the next model in that round."""
    config = Config(
        models=[ModelConfig(name="m1"), ModelConfig(name="m2")],
        discussion=DiscussionConfig(max_rounds=3),
        context=ContextConfig(mode="summary_only"),
    )
    session = Session(prompt="test", config=config.model_dump())
    
    # Simulate Round 1 being partially completed (only m1 responded)
    session.add_response("m1", "resp1", 1, 0)
    session.completed_rounds = 0 # Round 1 is not "completed"
    
    orchestrator = DiscussionOrchestrator(config=config, session=session)
    orchestrator.ollama.generate = AsyncMock(return_value=MagicMock(response="new resp"))
    orchestrator.session_manager.save = AsyncMock()
    
    await orchestrator.run()
    
    # The first call to generate should be for m2 in Round 1
    first_call_kwargs = orchestrator.ollama.generate.call_args_list[0].kwargs
    assert first_call_kwargs["model"] == "m2"
    # In Round 1, the prompt is the initial prompt, so we just check it contains the prompt
    assert "test" in first_call_kwargs.get("prompt", "")

@pytest.mark.asyncio
async def test_resume_already_completed_session():
    """Verify that running a completed session returns immediately."""
    config = Config(
        models=[ModelConfig(name="m1"), ModelConfig(name="m2")],
        discussion=DiscussionConfig(max_rounds=3),
    )
    session = Session(prompt="test", config=config.model_dump())
    session.status = "completed"
    
    orchestrator = DiscussionOrchestrator(config=config, session=session)
    orchestrator.ollama.generate = AsyncMock()
    
    await orchestrator.run()
    
    orchestrator.ollama.generate.assert_not_called()
