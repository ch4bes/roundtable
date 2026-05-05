# LLM Roundtable Discussion

A CLI/TUI application for running multi-model discussions with local LLMs via Ollama. Models respond to a prompt sequentially, consider each other's responses, and iterate until consensus is reached or max rounds are completed.

## Quick Start

```bash
# Install
pip install -e .

# Run the configuration wizard (select your models)
python scripts/auto_config.py

# Launch TUI
roundtable

# Or run a discussion from CLI
roundtable --prompt "What is the best approach to AI safety?"
```

## Features

- **Multi-Model Discussions**: Run discussions between multiple LLMs
- **Sequential Response**: Models respond in order, with configurable rotation each round
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

### Option 1: Virtual environment (recommended)

```bash
# Clone the repository
git clone https://github.com/ch4bes/roundtable.git
cd roundtable

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package in editable mode
python3 -m pip install -e .

# Run tests
python3 -m pytest

# Launch TUI
roundtable
```

To activate the virtual environment in the future:
```bash
cd roundtable
source venv/bin/activate
roundtable
```

### Option 2: User-wide installation (no activation needed)

If you prefer not to use a virtual environment:

```bash
# Clone the repository
git clone https://github.com/ch4bes/roundtable.git
cd roundtable

# Install the package in editable mode
python3 -m pip install -e . --user
```

**Note:** If `roundtable` command is not found after installation, you may need to add `~/.local/bin` to your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Add this to your `~/.zshrc` (macOS) or `~/.bashrc` (Linux) to make it permanent.

## Configuration

### Auto-Config Wizard

The easiest way to configure roundtable is to use the built-in configuration wizard:

```bash
python scripts/auto_config.py
```

The wizard guides you through 4 levels of configuration:

| Level | Description |
|-------|-------------|
| **Basic** | Select models and moderator |
| **Standard** | Basic + max rounds, consensus threshold, human participation |
| **Advanced** | Standard + context mode, discussion flow, final review |
| **All** | Everything - model parameters, consensus settings, storage, and more |

### Manual Configuration

Edit `config.json` to customize your setup:

```json
{
  "ollama": {
    "base_url": "http://localhost:11434",
    "timeout": 300
  },
  "models": [
    {"name": "qwen3.6:35b-a3b", "temperature": 0.7, "max_tokens": 2048, "num_ctx": 32768},
    {"name": "gemma4:26b", "temperature": 0.7, "max_tokens": 2048, "num_ctx": 32768},
    {"name": "qwen3.5:27b", "temperature": 0.7, "max_tokens": 2048, "num_ctx": 32768}
  ],
  "moderator": {
    "name": "qwen3.5:35b",
    "temperature": 0.5,
    "max_tokens": 2048,
    "num_ctx": 32768
  },
  "embeddings": {
    "model": "qwen3-embedding:8b"
  },
  "discussion": {
    "max_rounds": 6,
    "consensus_threshold": 0.75,
    "consensus_method": "clustering",
    "rotation_order": "fixed"
  },
  "context": {
    "mode": "summary_plus_last_n",
    "last_n_responses": 2,
    "response_preview_length": 800
  },
  "human_participant": {
    "enabled": true
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

# Combine prompt text with file content (joined with blank line)
roundtable --prompt "Discuss the following article:" --prompt-file article.txt
roundtable -p "Your question here" -f topic.txt  # short forms

# Include images in discussion (requires vision model like llava or llama3.2-vision)
roundtable -p "Describe this image" -i image.jpg
roundtable -i img1.png -i img2.jpg  # multiple images

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
- `H`: Add human response (during discussion)

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
└── tests/          # Test suite
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
"context_mode": "full"                 // All responses from all rounds
"context_mode": "summary_only"          // Only moderator summary (default)
"context_mode": "summary_plus_last_n"  // Summary + last N rounds of responses
```

When using `summary_plus_last_n`, you can also set:
```json
"last_n_responses": 2  // Number of recent rounds to include
```

### Rotation Strategies

Control the order models respond each round:

```json
"rotation_order": "sequential"  // Rotate through list each round
"rotation_order": "random"      // Shuffle order each round
"rotation_order": "fixed"        // Same order every round
```

### Human Participation

When human participation is enabled, humans can join the discussion:
- In CLI mode: Human is prompted after all models respond each round
- In TUI mode: Press `H` during a discussion to add a human response

Humans participate in consensus detection alongside the models.

## Testing

Run the test suite:

```bash
pytest
```

## License

MIT