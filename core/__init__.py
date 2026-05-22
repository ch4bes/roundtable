"""Public API exports for the roundtable project."""

from .config import Config
from .llm_client import LLMClient, GenerationResponse, EmbeddingResponse, ToolCall, ChatResponse
from .ollama_client import OllamaClient
from .discussion import DiscussionOrchestrator
from .similarity import SimilarityEngine
from .consensus import ConsensusDetector
from .tools import WebSearchTool, create_tool_executor, get_available_tools
from .exceptions import DimensionMismatchError

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
      "ConsensusDetector",
      "WebSearchTool",
      "create_tool_executor",
      "get_available_tools",
      "DimensionMismatchError",
]
