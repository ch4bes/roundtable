# Your Configured Models

## Current Configuration

Your roundtable is configured with these **3 Qwen models**:

| Model | Size | Role |
|-------|------|------|
| `qwen3-coder-next:cloud` | Cloud | Participant |
| `qwen3.5:27b` | 17 GB | Participant + Moderator |
| `qwen3:30b-thinking` | 18 GB | Participant |

## Configuration Details

- **Moderator**: qwen3.5:27b (temperature: 0.5)
- **Max Rounds**: 10
- **Consensus Threshold**: 0.85 (85% similarity)
- **Context Mode**: Summary only (models see moderator summary, not full transcript)

## Embedding Model Status

⚠️ **No embedding model detected**

The system will use **fallback text-based similarity** (Jaccard word overlap) for consensus detection. This works but is less accurate than embeddings.

### Optional: Install Embedding Model

For better semantic similarity detection:

```bash
ollama pull nomic-embed-text
```

This will automatically be used once installed (no config changes needed).

## Quick Start

```bash
# Launch TUI
roundtable

# Or start discussion from CLI
roundtable --prompt "What's your take on AI safety vs innovation speed?"

# Or use example prompt
roundtable --prompt-file example_prompt.txt
```

## Reconfigure Models

To use different models from your collection:

```bash
python scripts/auto_config.py
```

This will scan your Ollama models and select 3 diverse options.

## Your Available Models

You have 11 models total:
- qwen3.5 variants (9b, 27b, 35b, in BF16 and standard)
- qwen3 variants (30b-instruct, 30b-thinking)
- qwen3-coder variants (30b, next)
- Cloud models (qwen3.5:cloud, qwen3-coder-next:cloud)

The auto-config selected a good mix: cloud (fast), 27b (medium), 30b (large) for diverse perspectives.
