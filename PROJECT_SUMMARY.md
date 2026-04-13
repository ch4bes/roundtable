# Project Summary: LLM Roundtable Discussion

## Goal

Build an LLM roundtable discussion program where multiple local Ollama models can discuss a prompt sequentially, with a moderator summarizing responses after each round until consensus is reached or max rounds complete.

## Instructions

- Use local Ollama models (user has qwen family models)
- TUI interface with Textual framework
- JSON configuration
- Semantic similarity for consensus detection (with fallback to text-based Jaccard similarity)
- Save/resume sessions, export to MD or JSON

## Discoveries

1. **Timeout Issue**: The 120-second default timeout was too short for larger models. Increased to 300 seconds.

2. **Multiple Ollama Processes**: Initially saw concurrent processes. Changed moderator from qwen3.5:27b to qwen3.5:9b (different model from participants).

3. **Moderator Empty Response**: Debugging revealed that moderator model was returning 0 characters. Testing showed models work correctly via curl but had transient issues.

4. **Code Indentation Bugs**: During debugging, multiple indentation errors were introduced in `core/discussion.py` that broke the `DiscussionOrchestrator` class - methods like `run()`, `cleanup()`, `pause()`, `resume()`, `stop()`, `_generate_summary()`, and `_check_consensus()` were not being recognized as class methods because they were defined outside the class scope due to indentation issues.

5. **Embedding Model Missing**: User doesn't have `nomic-embed-text` installed, so similarity falls back to text-based Jaccard similarity.

## Accomplished

- Created complete project structure with core/, tui/, storage/, prompts/, tests/
- Configured with user's Ollama models: qwen3.5:35b, qwen3.5:27b, qwen3:30b-thinking as participants, qwen3.5:9b as moderator
- Increased timeout to 300s
- Added progress feedback prints for CLI
- Fixed massive indentation issues in core/discussion.py
- All 26 tests passing

## Relevant files / directories

- `/Users/james/Developer/OpenCode/roundtable/config.json` - Configuration with models, timeout, etc.
- `/Users/james/Developer/OpenCode/roundtable/core/discussion.py` - Main orchestrator (had indentation issues, now fixed)
- `/Users/james/Developer/OpenCode/roundtable/core/ollama_client.py` - Ollama API wrapper
- `/Users/james/Developer/OpenCode/roundtable/core/config.py` - Config loading
- `/Users/james/Developer/OpenCode/roundtable/core/similarity.py` - Similarity engine
- `/Users/james/Developer/OpenCode/roundtable/core/consensus.py` - Consensus detection
- `/Users/james/Developer/OpenCode/roundtable/tui/app.py` - TUI application
- `/Users/james/Developer/OpenCode/roundtable/tui/widgets.py` - TUI widgets
- `/Users/james/Developer/OpenCode/roundtable/tui/screens.py` - TUI screens
- `/Users/james/Developer/OpenCode/roundtable/storage/` - Session management and export
- `/Users/james/Developer/OpenCode/roundtable/prompts/system_prompts.py` - Moderator and participant prompts

## Discoveries (2026-04-12)

6. **NDJSON Streaming Bug**: The qwen3.5:9b model outputs in NDJSON streaming format even with `stream: false`. The Python client's `response.json()` only parsed the first line which had empty `response`. Fixed by manually parsing all lines.

7. **Thinking Field Output**: qwen3 models output content in `thinking` field instead of `response` when doing deep reasoning. The client now extracts from both fields.

8. **think Parameter Location**: The `think` parameter must be at the TOP LEVEL of the API payload, NOT inside the `options` dict. When inside `options`, it's silently ignored.

9. **Embedding Model**: User pulled `qwen3-embedding:8b` for semantic similarity (replacing unavailable `nomic-embed-text`).

10. **Prompt Engineering Issues**: Introduced syntax errors (smart quotes) while editing prompts - had to rewrite the file entirely.

## Accomplished (2026-04-12)

- Fixed NDJSON parsing in `core/ollama_client.py` (now aggregates all response chunks)
- Added `import json` to ollama_client.py
- Added thinking field extraction when response is empty
- Updated config.json to use `qwen3-embedding:8b` for embeddings
- Revised participant prompts to be concise (no greetings, hedging, rhetorical questions)
- Fixed moderator prompts for bullet-point summaries
- Added `"think": False` as top-level parameter to disable thinking output
- All 26 tests passing

## Relevant files / directories

- `/Users/james/Developer/OpenCode/roundtable/core/ollama_client.py` - Ollama API client with NDJSON and thinking field fixes
- `/Users/james/Developer/OpenCode/roundtable/prompts/system_prompts.py` - Moderator and participant prompts
- `/Users/james/Developer/OpenCode/roundtable/config.json` - Configuration with embedding model
- `/Users/james/Developer/OpenCode/roundtable/core/discussion.py` - Main orchestrator
- `/Users/james/Developer/OpenCode/roundtable/sessions/` - Saved session files

## How Consensus Works

After each round, the system calculates pairwise cosine similarity between all model responses using embeddings. For each unique pair of models, if similarity >= threshold (0.85), it's counted as an "agreeing pair". Consensus is reached when ALL pairs agree (100% of pairs above threshold).

Example: With 3 models, there are 3 pairs. If 2 pairs agree (66%), consensus is NOT reached (requires 3/3 = 100%).

In the last session (10 rounds), consensus was never reached despite models giving very similar answers. This suggests the 85% threshold may be too strict, or responses diverge slightly in wording despite semantic alignment.

## Next Steps

Run the discussion:
```bash
cd /Users/james/Developer/OpenCode/roundtable
python3 main.py --prompt "Why is the sky blue?"
```