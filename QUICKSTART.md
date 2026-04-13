# Quick Start Guide

## Prerequisites

1. **Install Ollama**: https://ollama.ai
2. **Pull required models**:
   ```bash
   ollama pull llama3.2
   ollama pull mistral
   ollama pull nomic-embed-text
   ```
3. **Start Ollama**:
   ```bash
   ollama serve
   ```

## Installation

```bash
cd roundtable
pip install -e .
```

## Usage

### Option 1: TUI Interface (Recommended)

```bash
roundtable
```

This launches the interactive terminal UI where you can:
- Enter discussion prompts
- Watch discussions in real-time
- Pause/resume/stop discussions
- Export results

### Option 2: CLI Mode

```bash
# Direct prompt
roundtable --prompt "What is the meaning of life?"

# From file
roundtable --prompt-file example_prompt.txt
```

### Check Connection

```bash
roundtable --check-ollama
```

## Configuration

Edit `config.json` to customize:

- **Models**: Add/remove LLMs for discussion
- **Moderator**: Choose which model summarizes
- **Embeddings**: Select embedding model for similarity
- **Thresholds**: Adjust consensus sensitivity
- **Max Rounds**: Limit discussion length

## Example Discussion

1. Launch TUI: `roundtable`
2. Press `Ctrl+P` to start
3. Enter prompt: "Should AI systems prioritize individual rights or collective welfare?"
4. Watch models discuss and iterate
5. Export results with `Ctrl+E`

## Troubleshooting

**"Ollama not available"**: Make sure `ollama serve` is running

**"Model not found"**: Run `ollama pull <model-name>`

**Slow responses**: Reduce `max_tokens` or use smaller models

**No consensus**: Lower `consensus_threshold` to 0.75
