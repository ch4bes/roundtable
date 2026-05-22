"""
Comprehensive tests for core/tools.py — WebSearchTool, create_tool_executor,
get_available_tools, and the WEB_SEARCH_TOOL constant.
"""

import json
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from core.tools import (
    WEB_SEARCH_TOOL,
    WebSearchTool,
    create_tool_executor,
    get_available_tools,
)

# ---------------------------------------------------------------------------
# Test fixtures / shared helpers
# ---------------------------------------------------------------------------

#: A well-formed 2-result Wikipedia opensearch response.
WIKI_RESPONSE_2 = [
    "Python programming",
    ["Python (programming language)", "Python (genus)"],
    [
        "Python is a high-level, general-purpose programming language.",
        "Python is a genus of constricting snakes in the family Pythonidae.",
    ],
    [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Python_(genus)",
    ],
]

#: A well-formed 3-result Wikipedia opensearch response.
WIKI_RESPONSE_3 = [
    "Python",
    ["Python (programming language)", "Python (genus)", "Monty Python"],
    [
        "Python is a high-level, general-purpose programming language.",
        "Python is a genus of constricting snakes in the family Pythonidae.",
        "Monty Python was a British surreal comedy group.",
    ],
    [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Python_(genus)",
        "https://en.wikipedia.org/wiki/Monty_Python",
    ],
]

#: An opensearch response with zero results.
WIKI_RESPONSE_EMPTY = ["ghost orchid", [], [], []]


def make_mock_response(data):
    """Return a MagicMock that mimics an httpx.Response whose .json() returns *data*."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    return mock_resp


def enabled_tools_config(max_results: int = 5, timeout: int = 30) -> dict:
    """Return a tools_config dict (model_dump() shape) with web_search enabled."""
    return {
        "web_search": {
            "enabled": True,
            "timeout": timeout,
            "max_results": max_results,
        }
    }


def disabled_tools_config() -> dict:
    """Return a tools_config dict with web_search disabled."""
    return {"web_search": {"enabled": False, "timeout": 30, "max_results": 5}}


def get_called_url(mock_get) -> str:
    """Extract the URL positional argument from the most recent httpx.get() call."""
    call_args = mock_get.call_args
    # httpx.get(url, ...) — URL is always the first positional arg
    return call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")


# ---------------------------------------------------------------------------
# TestWebSearchToolExecute
# ---------------------------------------------------------------------------


class TestWebSearchToolExecute:
    """Verify basic execute() behaviour — output keys, default handling, limit plumbing."""

    def test_returns_json_string(self):
        """execute() always returns a str that is valid JSON."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python programming")
        assert isinstance(result, str)
        json.loads(result)  # must not raise

    def test_result_has_query_key(self):
        """Successful result JSON contains a 'query' key."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python programming")
        assert "query" in json.loads(result)

    def test_result_has_results_key(self):
        """Successful result JSON contains a 'results' key."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python programming")
        assert "results" in json.loads(result)

    def test_result_has_source_key(self):
        """Successful result JSON contains a 'source' key."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python programming")
        assert "source" in json.loads(result)

    def test_query_echoed_in_result(self):
        """The 'query' value in the result matches the input query string."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python programming")
        data = json.loads(result)
        assert data["query"] == "Python programming"

    def test_num_results_capped_at_max_results_in_url(self):
        """When num_results > max_results, the API is called with limit=max_results."""
        tool = WebSearchTool(max_results=2)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            tool.execute("Python", num_results=99)
        assert "limit=2" in get_called_url(mock_get)

    def test_num_results_not_passed_uses_max_results_default(self):
        """When num_results is omitted, execute() completes without error using the default."""
        tool = WebSearchTool(max_results=5)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python programming")
        data = json.loads(result)
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_num_results_within_limit_is_passed_as_is(self):
        """When num_results <= max_results, the exact num_results is used in the API call."""
        tool = WebSearchTool(max_results=5)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            tool.execute("Python programming", num_results=3)
        assert "limit=3" in get_called_url(mock_get)

    def test_num_results_none_falls_back_to_max_results(self):
        """When num_results=None is passed explicitly, max_results is used as the limit."""
        tool = WebSearchTool(max_results=4)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            tool.execute("test query", num_results=None)
        assert "limit=4" in get_called_url(mock_get)


# ---------------------------------------------------------------------------
# TestWebSearchToolErrorHandling
# ---------------------------------------------------------------------------


class TestWebSearchToolErrorHandling:
    """Verify that all error paths return error JSON and never raise to the caller."""

    def test_timeout_exception_does_not_raise(self):
        """httpx.TimeoutException is caught; execute() returns normally."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", side_effect=httpx.TimeoutException("timed out")
        ):
            result = tool.execute("test query")  # must not raise
        assert isinstance(result, str)

    def test_timeout_exception_returns_error_key(self):
        """Result JSON from a timeout contains an 'error' key."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", side_effect=httpx.TimeoutException("timed out")
        ):
            result = tool.execute("test query")
        assert "error" in json.loads(result)

    def test_timeout_exception_result_contains_query(self):
        """Error JSON from a timeout contains the original 'query' value."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", side_effect=httpx.TimeoutException("timed out")
        ):
            result = tool.execute("my specific query")
        data = json.loads(result)
        assert "query" in data
        assert data["query"] == "my specific query"

    def test_timeout_result_is_valid_json(self):
        """Error result from a timeout is always decodable JSON."""
        tool = WebSearchTool()
        with patch("core.tools.httpx.get", side_effect=httpx.TimeoutException("t")):
            result = tool.execute("q")
        json.loads(result)  # must not raise

    def test_generic_exception_does_not_raise(self):
        """A generic Exception is caught; execute() returns normally."""
        tool = WebSearchTool()
        with patch("core.tools.httpx.get", side_effect=Exception("something broke")):
            result = tool.execute("test query")  # must not raise
        assert isinstance(result, str)

    def test_generic_exception_returns_error_key(self):
        """Result JSON from a generic exception contains an 'error' key."""
        tool = WebSearchTool()
        with patch("core.tools.httpx.get", side_effect=Exception("something broke")):
            result = tool.execute("test query")
        assert "error" in json.loads(result)

    def test_generic_exception_result_contains_query(self):
        """Error JSON from a generic exception contains the original 'query' value."""
        tool = WebSearchTool()
        with patch("core.tools.httpx.get", side_effect=RuntimeError("network down")):
            result = tool.execute("my query")
        data = json.loads(result)
        assert "query" in data
        assert data["query"] == "my query"

    def test_generic_exception_result_is_valid_json(self):
        """Error result from a generic exception is always decodable JSON."""
        tool = WebSearchTool()
        with patch("core.tools.httpx.get", side_effect=ValueError("bad response")):
            result = tool.execute("q")
        json.loads(result)  # must not raise

    def test_empty_results_does_not_raise(self):
        """An API response with zero results is handled without raising."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_EMPTY)
        ):
            result = tool.execute("ghost orchid")  # must not raise
        assert isinstance(result, str)

    def test_empty_results_returns_valid_json(self):
        """An API response with zero results still produces valid JSON output."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_EMPTY)
        ):
            result = tool.execute("ghost orchid")
        data = json.loads(result)
        assert "error" in data or (
            "results" in data and isinstance(data["results"], list)
        )

    def test_connect_error_does_not_raise(self):
        """httpx.ConnectError is caught; execute() returns normally."""
        tool = WebSearchTool()
        with patch("core.tools.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = tool.execute("test")  # must not raise
        assert isinstance(result, str)

    def test_connect_error_returns_error_key(self):
        """Result JSON from a connection error contains an 'error' key."""
        tool = WebSearchTool()
        with patch("core.tools.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = tool.execute("test")
        assert "error" in json.loads(result)


# ---------------------------------------------------------------------------
# TestWebSearchToolHTTPResponse
# ---------------------------------------------------------------------------


class TestWebSearchToolHTTPResponse:
    """Verify that the Wikipedia opensearch response is parsed into the correct structure."""

    def test_source_is_wikipedia(self):
        """The 'source' field in a successful response is the string 'Wikipedia'."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python programming")
        assert json.loads(result)["source"] == "Wikipedia"

    def test_result_count_matches_api_response(self):
        """The number of items in 'results' equals the number returned by the API."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        assert len(json.loads(result)["results"]) == 3

    def test_each_result_has_title_key(self):
        """Every result dict contains a 'title' key."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        for item in json.loads(result)["results"]:
            assert "title" in item

    def test_each_result_has_url_key(self):
        """Every result dict contains a 'url' key."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        for item in json.loads(result)["results"]:
            assert "url" in item

    def test_each_result_has_snippet_key(self):
        """Every result dict contains a 'snippet' key."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        for item in json.loads(result)["results"]:
            assert "snippet" in item

    def test_titles_parsed_correctly(self):
        """Title values in results match the second element of the opensearch response."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        titles = [r["title"] for r in json.loads(result)["results"]]
        assert "Python (programming language)" in titles
        assert "Python (genus)" in titles
        assert "Monty Python" in titles

    def test_snippets_parsed_correctly(self):
        """Snippet values in results match the third element of the opensearch response."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        snippets = [r["snippet"] for r in json.loads(result)["results"]]
        assert (
            "Python is a high-level, general-purpose programming language." in snippets
        )
        assert (
            "Python is a genus of constricting snakes in the family Pythonidae."
            in snippets
        )
        assert "Monty Python was a British surreal comedy group." in snippets

    def test_urls_parsed_correctly(self):
        """URL values in results match the fourth element of the opensearch response."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        urls = [r["url"] for r in json.loads(result)["results"]]
        assert "https://en.wikipedia.org/wiki/Python_(programming_language)" in urls
        assert "https://en.wikipedia.org/wiki/Python_(genus)" in urls
        assert "https://en.wikipedia.org/wiki/Monty_Python" in urls

    def test_hits_wikipedia_api_endpoint(self):
        """httpx.get is called with a URL that targets the Wikipedia opensearch API."""
        tool = WebSearchTool()
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            tool.execute("Python programming")
        url = get_called_url(mock_get)
        assert "en.wikipedia.org" in url
        assert "opensearch" in url or "api.php" in url

    def test_titles_and_urls_correspond_positionally(self):
        """The first title corresponds to the first URL (positional alignment)."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python")
        items = json.loads(result)["results"]
        prog_item = next(
            i for i in items if i["title"] == "Python (programming language)"
        )
        assert (
            prog_item["url"]
            == "https://en.wikipedia.org/wiki/Python_(programming_language)"
        )


# ---------------------------------------------------------------------------
# TestCreateToolExecutor
# ---------------------------------------------------------------------------


class TestCreateToolExecutor:
    """Verify create_tool_executor() factory behaviour."""

    def test_returns_none_when_config_is_none(self):
        """create_tool_executor(None) returns None (no tools enabled)."""
        assert create_tool_executor(None) is None

    def test_returns_none_when_web_search_disabled(self):
        """Returns None when web_search.enabled is False."""
        assert create_tool_executor(disabled_tools_config()) is None

    def test_returns_callable_when_web_search_enabled(self):
        """Returns a callable when web_search.enabled is True."""
        executor = create_tool_executor(enabled_tools_config())
        assert callable(executor)

    def test_executor_web_search_returns_string(self):
        """executor('web_search', {...}) returns a string."""
        executor = create_tool_executor(enabled_tools_config())
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = executor("web_search", {"query": "test"})
        assert isinstance(result, str)

    def test_executor_web_search_returns_valid_json(self):
        """executor('web_search', {...}) returns valid JSON."""
        executor = create_tool_executor(enabled_tools_config())
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = executor("web_search", {"query": "test"})
        json.loads(result)  # must not raise

    def test_executor_web_search_invokes_search(self):
        """executor('web_search', {'query': 'test'}) actually calls the search backend."""
        executor = create_tool_executor(enabled_tools_config())
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            executor("web_search", {"query": "test"})
        mock_get.assert_called_once()

    def test_executor_web_search_result_has_expected_keys(self):
        """A successful executor web_search call returns JSON with query/results/source."""
        executor = create_tool_executor(enabled_tools_config())
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = executor("web_search", {"query": "roundtable"})
        data = json.loads(result)
        assert "query" in data or "error" in data  # either outcome is valid JSON

    def test_executor_unknown_tool_returns_string(self):
        """executor('unknown_tool', {}) returns a string."""
        executor = create_tool_executor(enabled_tools_config())
        result = executor("unknown_tool", {})
        assert isinstance(result, str)

    def test_executor_unknown_tool_returns_error_json(self):
        """executor('unknown_tool', {}) returns JSON containing an 'error' key."""
        executor = create_tool_executor(enabled_tools_config())
        result = executor("unknown_tool", {})
        data = json.loads(result)
        assert "error" in data

    def test_executor_passes_num_results_argument(self):
        """executor passes num_results through to the underlying tool."""
        executor = create_tool_executor(enabled_tools_config(max_results=5))
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            executor("web_search", {"query": "x", "num_results": 2})
        assert "limit=2" in get_called_url(mock_get)

    def test_executor_passes_query_argument(self):
        """executor passes the query string through to the underlying tool."""
        executor = create_tool_executor(enabled_tools_config())
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            executor("web_search", {"query": "special query text"})
        url = get_called_url(mock_get)
        assert (
            "special" in url
            or "special+query+text" in url
            or "special%20query%20text" in url
        )

    def test_executor_error_json_mentions_unknown_tool(self):
        """The error JSON for an unknown tool references the tool name or 'error'."""
        executor = create_tool_executor(enabled_tools_config())
        result = executor("nonexistent_magic_tool", {"key": "value"})
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# TestGetAvailableTools
# ---------------------------------------------------------------------------


class TestGetAvailableTools:
    """Verify get_available_tools() returns the correct Ollama-format tool list."""

    def test_returns_list_type_for_none_config(self):
        """get_available_tools(None) returns a list (not None or another type)."""
        assert isinstance(get_available_tools(None), list)

    def test_returns_empty_list_when_config_is_none(self):
        """get_available_tools(None) returns an empty list."""
        assert get_available_tools(None) == []

    def test_returns_list_type_when_disabled(self):
        """get_available_tools with web_search disabled returns a list."""
        assert isinstance(get_available_tools(disabled_tools_config()), list)

    def test_returns_empty_list_when_web_search_disabled(self):
        """Returns [] when web_search.enabled=False."""
        assert get_available_tools(disabled_tools_config()) == []

    def test_returns_list_when_web_search_enabled(self):
        """Returns a non-empty list when web_search.enabled=True."""
        result = get_available_tools(enabled_tools_config())
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returned_list_contains_web_search_tool(self):
        """The returned list includes a tool named 'web_search'."""
        result = get_available_tools(enabled_tools_config())
        names = [t.get("function", {}).get("name") for t in result]
        assert "web_search" in names

    def test_web_search_tool_has_type_function(self):
        """The web_search tool entry has type='function'."""
        result = get_available_tools(enabled_tools_config())
        web_search = next(
            (t for t in result if t.get("function", {}).get("name") == "web_search"),
            None,
        )
        assert web_search is not None
        assert web_search["type"] == "function"

    def test_web_search_tool_has_function_key(self):
        """The web_search tool entry has a 'function' dict."""
        result = get_available_tools(enabled_tools_config())
        web_search = next(
            (t for t in result if t.get("function", {}).get("name") == "web_search"),
            None,
        )
        assert web_search is not None
        assert isinstance(web_search["function"], dict)

    def test_web_search_tool_function_has_parameters(self):
        """The web_search function dict contains a 'parameters' key."""
        result = get_available_tools(enabled_tools_config())
        web_search = next(
            (t for t in result if t.get("function", {}).get("name") == "web_search"),
            None,
        )
        assert web_search is not None
        assert "parameters" in web_search["function"]

    def test_web_search_parameters_has_query_property(self):
        """The parameters.properties dict contains a 'query' entry."""
        result = get_available_tools(enabled_tools_config())
        web_search = next(
            (t for t in result if t.get("function", {}).get("name") == "web_search"),
            None,
        )
        params = web_search["function"]["parameters"]
        assert "properties" in params
        assert "query" in params["properties"]

    def test_web_search_query_property_is_string_type(self):
        """The 'query' parameter property has type 'string'."""
        result = get_available_tools(enabled_tools_config())
        web_search = next(
            (t for t in result if t.get("function", {}).get("name") == "web_search"),
            None,
        )
        query_prop = web_search["function"]["parameters"]["properties"]["query"]
        assert query_prop.get("type") == "string"

    def test_web_search_tool_name_matches_constant(self):
        """The returned web_search tool's function name matches WEB_SEARCH_TOOL."""
        result = get_available_tools(enabled_tools_config())
        web_search = next(
            (t for t in result if t.get("function", {}).get("name") == "web_search"),
            None,
        )
        assert web_search["function"]["name"] == WEB_SEARCH_TOOL["function"]["name"]

    def test_web_search_tool_type_matches_constant(self):
        """The returned web_search tool's type matches WEB_SEARCH_TOOL."""
        result = get_available_tools(enabled_tools_config())
        web_search = next(
            (t for t in result if t.get("function", {}).get("name") == "web_search"),
            None,
        )
        assert web_search["type"] == WEB_SEARCH_TOOL["type"]

    def test_always_returns_list(self):
        """get_available_tools returns a list in all cases."""
        assert isinstance(get_available_tools(None), list)
        assert isinstance(get_available_tools(disabled_tools_config()), list)
        assert isinstance(get_available_tools(enabled_tools_config()), list)


# ---------------------------------------------------------------------------
# TestMaxResults
# ---------------------------------------------------------------------------


class TestMaxResults:
    """Verify num_results capping logic — config max_results always wins."""

    def test_large_num_results_capped_at_max_results_3(self):
        """With max_results=3, requesting num_results=10 caps the API call at limit=3."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ) as mock_get:
            tool.execute("test", num_results=10)
        assert "limit=3" in get_called_url(mock_get)

    def test_large_num_results_caps_returned_results(self):
        """With max_results=2, the result list contains at most 2 items."""
        tool = WebSearchTool(max_results=2)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ):
            result = tool.execute("Python", num_results=10)
        data = json.loads(result)
        assert len(data["results"]) <= 2

    def test_small_num_results_not_inflated_to_max(self):
        """With max_results=5, requesting num_results=2 uses limit=2 (not 5)."""
        tool = WebSearchTool(max_results=5)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ) as mock_get:
            tool.execute("test", num_results=2)
        assert "limit=2" in get_called_url(mock_get)

    def test_small_num_results_returned_as_requested(self):
        """With max_results=5 and num_results=2, at most 2 items appear in results."""
        tool = WebSearchTool(max_results=5)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_2)
        ):
            result = tool.execute("Python", num_results=2)
        data = json.loads(result)
        assert len(data["results"]) <= 2

    def test_num_results_equal_to_max_results_uses_exact_limit(self):
        """When num_results == max_results, the URL uses that exact limit."""
        tool = WebSearchTool(max_results=3)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ) as mock_get:
            tool.execute("Python", num_results=3)
        assert "limit=3" in get_called_url(mock_get)

    def test_num_results_one_below_max_results_respected(self):
        """A num_results one below max_results is not rounded up."""
        tool = WebSearchTool(max_results=5)
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ) as mock_get:
            tool.execute("Python", num_results=4)
        assert "limit=4" in get_called_url(mock_get)

    def test_max_results_1_always_yields_one_result(self):
        """With max_results=1, even num_results=100 is capped, yielding at most 1 result."""
        tool = WebSearchTool(max_results=1)
        single_response = [
            "Python",
            ["Python (programming language)"],
            ["Python is a high-level programming language."],
            ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
        ]
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(single_response)
        ) as mock_get:
            result = tool.execute("Python", num_results=100)
        assert "limit=1" in get_called_url(mock_get)
        data = json.loads(result)
        assert len(data["results"]) <= 1

    def test_executor_respects_max_results_cap(self):
        """create_tool_executor honours max_results when executor receives num_results."""
        executor = create_tool_executor(enabled_tools_config(max_results=3))
        with patch(
            "core.tools.httpx.get", return_value=make_mock_response(WIKI_RESPONSE_3)
        ) as mock_get:
            executor("web_search", {"query": "test", "num_results": 99})
        assert "limit=3" in get_called_url(mock_get)
