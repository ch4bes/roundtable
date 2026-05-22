"""
Tool definitions and executors for Ollama function calling.

This module provides tools that models can use during discussions,
such as web search for fact-checking.
"""

import json
import logging
import urllib.parse
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas (Ollama function-calling format)
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search Wikipedia for factual information. "
            "Use this to verify claims, look up statistics, dates, names, or "
            "other facts that arise during the discussion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — be specific and include key terms",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------


class WebSearchTool:
    """Web search via the Wikipedia OpenSearch API.

    Uses ``httpx`` (a project dependency) instead of shelling out to ``curl``
    so the tool is portable, testable, and compatible with async hosts.
    """

    def __init__(self, timeout: int = 30, max_results: int = 5) -> None:
        self.timeout = timeout
        # Server-side cap: even if the model requests more, we never fetch more
        # than this many results, keeping response sizes manageable.
        self.max_results = max_results

    def execute(self, query: str, num_results: int | None = None) -> str:
        """Run the search and return a JSON string.

        Args:
            query: The search query.
            num_results: How many results the caller wants.  Capped at
                ``self.max_results``; defaults to ``self.max_results`` when
                omitted or ``None``.

        Returns:
            JSON string with keys ``query``, ``results`` (list of
            ``{title, url, snippet}``), and ``source``, **or** a JSON error
            object on failure.
        """
        n = min(
            num_results if num_results is not None else self.max_results,
            self.max_results,
        )
        return self._search_wikipedia(query, n)

    def _search_wikipedia(self, query: str, num_results: int) -> str:
        """Query the Wikipedia OpenSearch API and format results."""
        encoded_query = urllib.parse.quote_plus(query)
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=opensearch&search={encoded_query}"
            f"&limit={num_results}&format=json"
        )

        try:
            response = httpx.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            logger.warning("Wikipedia search timed out for query: %r", query)
            return json.dumps({"error": "Wikipedia search timed out", "query": query})
        except Exception as exc:
            logger.warning("Wikipedia search failed for query %r: %s", query, exc)
            return json.dumps(
                {"error": f"Wikipedia search failed: {exc}", "query": query}
            )

        # OpenSearch returns [query, [titles], [descriptions], [urls]]
        if not (isinstance(data, list) and len(data) >= 2 and data[1]):
            return json.dumps({"error": "No Wikipedia results found", "query": query})

        titles: list[str] = data[1]
        descriptions: list[str] = data[2] if len(data) >= 3 else []
        urls: list[str] = data[3] if len(data) >= 4 else []

        results = [
            {
                "title": titles[i],
                "url": urls[i] if i < len(urls) else "",
                "snippet": descriptions[i] if i < len(descriptions) else "",
            }
            for i in range(min(num_results, len(titles)))
        ]

        return json.dumps({"query": query, "results": results, "source": "Wikipedia"})


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_tool_executor(
    tools_config: dict | None = None,
) -> Callable[[str, dict], str] | None:
    """Return a ``(tool_name, arguments) -> result_str`` callable, or ``None``.

    Args:
        tools_config: The ``tools`` section of ``Config.model_dump()``.

    Returns:
        A synchronous executor function when at least one tool is enabled,
        otherwise ``None``.
    """
    if not tools_config:
        return None

    executors: dict[str, WebSearchTool] = {}

    ws_cfg = tools_config.get("web_search", {})
    if ws_cfg.get("enabled", False):
        executors["web_search"] = WebSearchTool(
            timeout=ws_cfg.get("timeout", 30),
            max_results=ws_cfg.get("max_results", 5),
        )

    if not executors:
        return None

    def executor(tool_name: str, arguments: dict) -> str:
        tool = executors.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name!r}"})
        try:
            return tool.execute(**arguments)
        except Exception as exc:
            logger.warning("Tool %r raised an error: %s", tool_name, exc)
            return json.dumps({"error": str(exc)})

    return executor


def get_available_tools(tools_config: dict | None = None) -> list[dict]:
    """Return the list of Ollama-format tool definitions for enabled tools.

    Args:
        tools_config: The ``tools`` section of ``Config.model_dump()``.
    """
    if not tools_config:
        return []

    tools: list[dict] = []
    ws_cfg = tools_config.get("web_search", {})
    if ws_cfg.get("enabled", False):
        tools.append(WEB_SEARCH_TOOL)

    return tools
