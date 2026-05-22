"""Public API exports for the roundtable project."""

from .config import Config
from .consensus import ConsensusDetector
from .discussion import DiscussionOrchestrator
from .exceptions import DimensionMismatchError
from .llm_client import (
    ChatResponse,
    EmbeddingResponse,
    GenerationResponse,
    LLMClient,
    ToolCall,
)
from .ollama_client import OllamaClient
from .similarity import SimilarityEngine
from .summary_parser import SummaryParser
from .tools import WebSearchTool, create_tool_executor, get_available_tools

__all__ = [
    "Config",
    "LLMClient",
    "GenerationResponse",
    "EmbeddingResponse",
    "ToolCall",
    "ChatResponse",
    "OllamaClient",
    "DiscussionOrchestrator",
    "SimilarityEngine",
    "SummaryParser",
    "ConsensusDetector",
    "WebSearchTool",
    "create_tool_executor",
    "get_available_tools",
    "DimensionMismatchError",
]
