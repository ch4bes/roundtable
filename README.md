# LLM Roundtable Discussion

A CLI/TUI application for running multi-model discussions with local LLMs via Ollama. Models respond to a prompt sequentially, consider each other's responses, and iterate until consensus is reached or max rounds are completed.

## Features

- **Multi-Model Discussions**: Run discussions between multiple LLMs
- **Sequential Response**: Models respond in order, with rotation each round
- **Consensus Detection**: Semantic similarity-based consensus using embeddings
- **Moderator Summary**: LLM-generated summaries after each round
- **TUI Interface**: Live terminal UI with real-time updates
- **Save/Resume**: Auto-save sessions and resume later
- **Export**: Export discussions to Markdown or JSON
- **Fully Local**: Works with Ollama for complete privacy

## Requirements

- Python 3.10+
- Ollama running locally (http://localhost:11434)
- At least 2 LLM models installed in Ollama
- Embedding model recommended (e.g., `qwen3-embedding:8b`) for better similarity detection
  - Without embeddings, falls back to Jaccard similarity (word overlap)

## Installation

```bash
# Clone the repository
cd roundtable

# Run tests
pytest

# Launch TUI
roundtable
```

## Configuration

Edit `config.json` to customize your setup:

```json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "timeout": 300
  },
  "models": [
    {"name": "qwen3.6:35b-a3b", "temperature": 0.7, "max_tokens": 2048, "num_ctx": 32768},
    {"name": "qwen3.5:35b", "temperature": 0.7, "max_tokens": 2048, "num_ctx": 32768},
    {"name": "gemma4:31b", "temperature": 0.7, "max_tokens": 2048, "num_ctx": 32768}
  ],
  "moderator": {
    "name": "qwen3.6:35b-a3b",
    "temperature": 0.5,
    "max_tokens": 2048,
    "num_ctx": 32768
  },
  "embeddings": {
    "model": "qwen3-embedding:8b"
  },
  "discussion": {
    "max_rounds": 10,
    "consensus_threshold": 0.75
  }
}
```

## Usage

### Launch TUI

```bash
roundtable
```

### CLI Mode

```bash
# Start discussion with prompt
roundtable --prompt "What is the meaning of life?"

# Start from file
roundtable -f discussion.txt

# Force TUI mode
roundtable --tui

# Check Ollama connection
roundtable --check-ollama

# List saved sessions
roundtable --list-sessions

# Export a session
roundtable --export SESSION_ID --export-format md

# Use custom config
roundtable -c /path/to/config.json
```

### TUI Controls

- `Ctrl+P`: Start new discussion
- `Ctrl+S`: Save session
- `Ctrl+E`: Export session
- `Ctrl+O`: Open saved session
- `Space`: Pause/Resume
- `Ctrl+Q`: Stop discussion
- `Ctrl+C`: Quit

## How It Works

1. **Initial Response**: Each model provides their initial thoughts on the prompt
2. **Summary**: Moderator model summarizes all responses, highlighting agreements/disagreements
3. **Updated Response**: Models consider the summary and provide updated responses
4. **Consensus Check**: Semantic similarity is calculated between all responses
5. **Repeat**: Steps 2-4 continue until consensus or max rounds

### Consensus Detection

- **With embeddings**: Uses embedding models to vectorize responses, calculates cosine similarity
- **Without embeddings**: Falls back to Jaccard similarity (word overlap analysis)
- Consensus reached when all pairs exceed threshold
- Progress shown as percentage in TUI

## Project Structure

```
roundtable/
├── core/           # Core logic (discussion, similarity, consensus, config)
├── tui/            # Terminal UI components
├── storage/        # Session management and export
├── prompts/        # Prompt templates
├── scripts/        # Utility scripts (auto_config, update_config)
├── sessions/       # Saved discussions
└── tests/          # Test suite (153 tests)
```

## Examples

### Example Discussion Topics

- Ethical dilemmas and moral reasoning
- Technical architecture decisions
- Creative brainstorming sessions
- Problem-solving with multiple perspectives
- Code review and best practices

### Sample Output

See exported Markdown files in `sessions/` directory for examples.

## Troubleshooting

### Ollama Not Available

```bash
# Start Ollama
ollama serve

# Check connection
roundtable --check-ollama
```

### Models Not Found

```bash
# Install required models
ollama pull [model]
```

### Consensus Never Reached

- Lower `consensus_threshold` in config (e.g., 0.75)
- Increase `max_rounds` for more discussion time
- Use more similar models (e.g., different versions of Qwen)

## Advanced Configuration

### Context Modes

Configure how much history is included in each model's context:

```json
"context_mode": "full"        // All responses and summaries
"context_mode": "last_summaries"  // Only recent summaries
"context_mode": "last_round"  // Only current round
"context_mode": "compact"     // Compressed summary
```

### Rotation Strategies

Control the order models respond each round:

```json
"rotation_strategy": "sequential"  // Rotate in list order
"rotation_strategy": "random"      // Random shuffle each round
"rotation_strategy": "fixed"       // Always same order
```

### Human Participation

In TUI mode, press `H` during a discussion to add a human response to the round. Humans can participate in consensus detection.

## Testing

Run the test suite:

```bash
pytest
```

153 tests covering core modules, prompts, session management, export, and more.

## License

MIT
