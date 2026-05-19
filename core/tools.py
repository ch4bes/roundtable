"""
Tool definitions and executors for Ollama function calling.

This module provides tools that models can use during discussions,
such as web search for fact-checking.
"""

import json
import subprocess
import urllib.parse
from typing import Callable


# Web Search Tool Definition (Ollama format)
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search Wikipedia for factual information. "
                       "Use this when you need to verify claims, look up facts, "
                       "statistics, or dates during discussions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query - be specific and include key terms"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}


class WebSearchTool:
    """
    Web search tool using curl to query Wikipedia API.
    
    No API keys or hosting required - uses Wikipedia exclusively.
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def execute(self, query: str, num_results: int = 5) -> str:
        """
        Execute a web search using curl.
        
        :param query: Search query
        :param num_results: Number of results to return
        :returns: JSON string with search results
        """
        return self._search_wikipedia(query, num_results)
    
    def _search_wikipedia(self, query: str, num_results: int) -> str:
        """Search Wikipedia API using curl."""
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit={num_results}&format=json"
            
            result = subprocess.run(
                ["curl", "-s", "-m", str(self.timeout), url],
                capture_output=True,
                text=True,
                timeout=self.timeout + 5
            )
            
            if result.returncode != 0:
                return json.dumps({"error": "Wikipedia search failed", "query": query})
            
            data = json.loads(result.stdout)
            
            if len(data) >= 2 and len(data[1]) > 0:
                titles = data[1]
                descriptions = data[2] if len(data) >= 3 else []
                urls = data[3] if len(data) >= 4 else []
                
                results = []
                for i in range(min(num_results, len(titles))):
                    results.append({
                        "title": titles[i],
                        "url": urls[i] if i < len(urls) else "",
                        "snippet": descriptions[i] if i < len(descriptions) else "",
                    })
                
                return json.dumps({
                    "query": query,
                    "results": results,
                    "source": "Wikipedia",
                })
            
            return json.dumps({"error": "No Wikipedia results", "query": query})
            
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Wikipedia search timed out", "query": query})
        except Exception as e:
            return json.dumps({"error": f"Wikipedia search failed: {str(e)}", "query": query})


def create_tool_executor(tools_config: dict | None = None) -> Callable[[str, dict], str] | None:
    """
    Create a tool executor function based on configuration.
    
    :param tools_config: Configuration for tools (e.g., {"web_search": {...}})
    :returns: Function that executes tool calls (name, arguments) -> result string
    """
    if not tools_config:
        return None
    
    executors = {}
    
    # Web search tool
    ws_config = tools_config.get("web_search", {})
    if ws_config.get("enabled", False):
        executors["web_search"] = WebSearchTool(
            timeout=ws_config.get("timeout", 30),
        )
    
    if not executors:
        return None
    
    def executor(tool_name: str, arguments: dict) -> str:
        if tool_name in executors:
            tool = executors[tool_name]
            if hasattr(tool, "execute"):
                return tool.execute(**arguments)
            return json.dumps({"error": f"Tool {tool_name} has no execute method"})
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    return executor


# Tool list for passing to Ollama
def get_available_tools(tools_config: dict | None = None) -> list[dict]:
    """
    Get list of available tool definitions for Ollama.
    
    :param tools_config: Configuration for tools
    :returns: List of tool definitions in Ollama format
    """
    tools = []
    
    if not tools_config:
        return tools
    
    ws_config = tools_config.get("web_search", {})
    if ws_config.get("enabled", False):
        tools.append(WEB_SEARCH_TOOL)
    
    return tools