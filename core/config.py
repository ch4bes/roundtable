import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout: int = 120


class ModelConfig(BaseModel):
    name: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    num_ctx: int = Field(default=8192, gt=0)


class EmbeddingsConfig(BaseModel):
    model: str = "nomic-embed-text"


class ConsensusConfig(BaseModel):
    mode: Literal["moderator_decides", "programmatic_decides"] = "moderator_decides"
    threshold: float = Field(default=0.75, ge=0, le=1)
    method: Literal["pairwise", "clustering"] = "clustering"
    strictness: Literal["full", "main_point"] = "main_point"


class DiscussionConfig(BaseModel):
    max_rounds: int = Field(default=10, gt=0)
    consensus_threshold: float = Field(default=0.75, ge=0, le=1)
    consensus_method: Literal["pairwise", "clustering"] = "clustering"
    rotation_order: Literal["sequential", "random"] = "sequential"
    final_review_enabled: bool = True
    
    # Embedding-based similarity thresholds for _check_consensus:
    consensus_agreement_when_reached: float = Field(
        default=0.50, ge=0, le=1,
        description="Similarity threshold when moderator assessed REACHED (lower bar)"
    )
    consensus_agreement_when_not_reached: float = Field(
        default=0.75, ge=0, le=1,
        description="Similarity threshold when moderator assessed NOT REACHED (higher bar)"
    )
    reprompt_agreement_threshold: float = Field(
        default=0.70, ge=0, le=1,
        description="Agreement pct (0-1) at which moderator is reprompted to reconsider"
    )


class ContextConfig(BaseModel):
    mode: Literal["full", "summary_only", "summary_plus_last_n"] = "summary_only"
    last_n_responses: int = Field(default=2, gt=0)


class StorageConfig(BaseModel):
    sessions_dir: str = "./sessions"
    auto_save: bool = True
    export_format: Literal["md", "json"] = "md"


class HumanParticipantConfig(BaseModel):
    enabled: bool = False
    prompt: str = "Share your perspective on: {prompt}"
    display_name: str = "Human"


class Config(BaseSettings):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: list[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(name="qwen3.6:35b-a3b"),
            ModelConfig(name="qwen3.5:35b"),
            ModelConfig(name="qwen3.5:27b"),
            ModelConfig(name="qwen3:30b-thinking"),
            ModelConfig(name="gemma4:31b"),
            ModelConfig(name="gemma4:26b"),
            ModelConfig(name="gemma4:e4b"),
        ]
    )
    moderator: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            name="qwen3.6:35b-a3b", temperature=0.5, max_tokens=2048
        )
    )
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    human_participant: HumanParticipantConfig = Field(default_factory=HumanParticipantConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    discussion: DiscussionConfig = Field(default_factory=DiscussionConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    default_prompt: str = ""

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.json"
        else:
            config_path = Path(config_path)

        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return cls(**config_data)
        return cls()

    def save(self, config_path: str | Path) -> None:
        config_path = Path(config_path)
        with open(config_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 models are required for a roundtable discussion")
        return v
