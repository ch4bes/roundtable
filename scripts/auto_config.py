#!/usr/bin/env python3
"""Automatically configure roundtable with your best available Ollama models."""

import json
import subprocess
import sys
from pathlib import Path


CODE_BLOCK_OPEN = "```bash"
CODE_BLOCK_CLOSE = "```"


def get_ollama_models():
    """Get list of available Ollama models with sizes."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")[1:]
        models = []
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    size_str = parts[2]
                    size_gb = parse_size(size_str)
                    models.append({"name": name, "size_gb": size_gb})
        return models
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return []


def parse_size(size_str: str) -> float:
    """Parse size string like '18 GB' to float."""
    try:
        if "GB" in size_str:
            return float(size_str.replace("GB", "").strip())
        elif "MB" in size_str:
            return float(size_str.replace("MB", "").strip()) / 1024
        elif size_str == "-":
            return 0
        return float(size_str)
    except:
        return 0


def select_diverse_models(models: list[dict], count: int = 3) -> list[dict]:
    """Select models of different sizes for diversity."""
    if len(models) <= count:
        return models

    sorted_models = sorted(models, key=lambda m: m["size_gb"])

    selected = []
    step = max(1, len(sorted_models) // count)

    for i in range(0, len(sorted_models), step):
        if len(selected) >= count:
            break
        selected.append(sorted_models[i])

    while len(selected) < count:
        selected.append(sorted_models[len(selected)])

    return selected[:count]


def has_embedding_model(models: list[dict]) -> str | None:
    """Check if an embedding model is available."""
    embedding_keywords = ["embed", "embedding"]
    for model in models:
        name_lower = model["name"].lower()
        if any(kw in name_lower for kw in embedding_keywords):
            return model["name"]
    return None


def prompt_with_default(prompt_text: str, default: int) -> int:
    """Prompt user with a default value."""
    user_input = input(f"{prompt_text} (default: {default}): ").strip()
    if user_input == "":
        return default
    try:
        return int(user_input)
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def prompt_float_with_default(prompt_text: str, default: float) -> float:
    """Prompt user with a default float value."""
    user_input = input(f"{prompt_text} (default: {default}): ").strip()
    if user_input == "":
        return default
    try:
        return float(user_input)
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def select_moderator(models: list[dict]) -> dict:
    """Let user select a moderator from the available models."""
    print("\nAvailable models for moderator selection:")
    for i, model in enumerate(models, 1):
        size = f"{model['size_gb']:.1f} GB" if model["size_gb"] > 0 else "cloud"
        print(f"  {i}. {model['name']} ({size})")
    
    while True:
        user_input = input(f"\nWhich model should be moderator? (1-{len(models)}): ").strip()
        if user_input == "":
            print(f"  Defaulting to: {models[0]['name']}")
            return models[0]
        try:
            choice = int(user_input)
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print(f"  Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("  Please enter a number from the list.")


def update_config():
    """Auto-configure roundtable with best available models."""
    print("Scanning Ollama models...")
    all_models = get_ollama_models()

    if not all_models:
        print("✗ No models found. Make sure Ollama is running.")
        return False

    print(f"✓ Found {len(all_models)} models\n")

    # Interactive prompts for key configuration
    num_models = prompt_with_default("How many models to use?", 3)
    num_models = min(num_models, len(all_models))
    
    max_rounds = prompt_with_default("Max discussion rounds?", 10)
    consensus_threshold = prompt_float_with_default("Consensus threshold?", 0.75)
    timeout = prompt_with_default("Ollama timeout in seconds?", 300)

    # Select diverse models
    selected = select_diverse_models(all_models, count=num_models)
    moderator = select_moderator(selected)
    embedding_model = has_embedding_model(all_models)

    print("\n" + "=" * 50)
    print("Selected models for roundtable:")
    print("=" * 50)
    for i, model in enumerate(selected, 1):
        role = "Moderator" if model["name"] == moderator["name"] else "Participant"
        size = f"{model['size_gb']:.1f} GB" if model["size_gb"] > 0 else "cloud"
        print(f"  {i}. {model['name']} ({size}) - {role}")

    if embedding_model:
        print(f"\n✓ Embedding model found: {embedding_model}")
    else:
        print("\n⚠ No embedding model found")
        print("  Install with: ollama pull qwen3-embedding:8b")
        print("  (Will use fallback text-based similarity)")

    config = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "timeout": timeout
        },
        "models": [
            {"name": m["name"], "temperature": 0.7, "max_tokens": 2048, "num_ctx": 32768}
            for m in selected
        ],
        "moderator": {
            "name": moderator["name"],
            "temperature": 0.5,
            "max_tokens": 2048,
            "num_ctx": 32768
        },
        "discussion": {
            "max_rounds": max_rounds,
            "consensus_threshold": consensus_threshold,
            "consensus_method": "clustering",
            "rotation_order": "sequential"
        },
        "context": {
            "mode": "summary_only",
            "last_n_responses": 2
        },
        "storage": {
            "sessions_dir": "./sessions",
            "auto_save": True,
            "export_format": "md"
        },
        "default_prompt": "",
        "embeddings": {
            "model": embedding_model if embedding_model else "qwen3-embedding:8b"
        },
        "human_participant": {
            "enabled": True,
            "prompt": "Share your perspective on: {prompt}",
            "display_name": "Human"
        },
        "consensus": {
            "mode": "moderator_decides",
            "threshold": 0.75,
            "method": "clustering"
        }
    }

    config_path = Path("config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Generate CONFIGURED_MODELS.md
    generate_configured_models_md(selected, moderator, 0.5, embedding_model, max_rounds, consensus_threshold)

    print(f"\n✓ Updated {config_path}")
    print("\nReady to start! Run: roundtable")

    return True


def generate_configured_models_md(selected: list[dict], moderator: dict,
                                   moderator_temp: float,
                                   embedding_model: str | None, max_rounds: int,
                                   consensus_threshold: float) -> None:
    """Generate CONFIGURED_MODELS.md based on current configuration."""
    moderator_name = moderator["name"]

    lines = []
    lines.append("# Your Configured Models")
    lines.append("")
    lines.append("## Current Configuration")
    lines.append("")

    model_count = len(selected)
    lines.append(f"Your roundtable is configured with these **{model_count} models**:\n")

    lines.append("| Model | Size | Role |")
    lines.append("|-------|------|------|")
    for model in selected:
        size = f"{model['size_gb']:.1f} GB" if model['size_gb'] > 0 else "Cloud"
        role = "Moderator" if model['name'] == moderator_name else "Participant"
        lines.append(f"| `{model['name']}` | {size} | {role} |")

    lines.append("")
    lines.append("## Configuration Details")
    lines.append("")
    lines.append(f"- **Moderator**: {moderator_name} (temperature: {moderator_temp})")
    lines.append(f"- **Max Rounds**: {max_rounds}")
    lines.append(f"- **Consensus Threshold**: {consensus_threshold} ({int(consensus_threshold * 100)}% similarity)")
    lines.append("- **Context Mode**: Summary only (models see moderator summary, not full transcript)")
    lines.append("- **Consensus Method**: Clustering")
    lines.append("- **Human Participant**: Enabled")
    lines.append("")

    lines.append("## Embedding Model Status")
    lines.append("")
    if embedding_model:
        lines.append(f"**Embedding model**: {embedding_model}")
    else:
        lines.append("⚠️ **No embedding model detected**")
        lines.append("")
        lines.append("The system will use **fallback text-based similarity** (Jaccard word overlap) for consensus detection. This works but is less accurate than embeddings.")
        lines.append("")
        lines.append("## Optional: Install Embedding Model")
        lines.append("")
        lines.append("For better semantic similarity detection:")
        lines.append("")
        lines.append(CODE_BLOCK_OPEN)
        lines.append("ollama pull qwen3-embedding:8b")
        lines.append(CODE_BLOCK_CLOSE)
    lines.append("")
    lines.append("## Quick Start")
    lines.append("")
    lines.append(CODE_BLOCK_OPEN)
    lines.append("# Launch TUI")
    lines.append("roundtable")
    lines.append("")
    lines.append("# Or start discussion from CLI")
    lines.append('roundtable --prompt "What\'s your take on AI safety vs innovation speed?"')
    lines.append(CODE_BLOCK_CLOSE)
    lines.append("")
    lines.append("## Reconfigure Models")
    lines.append("")
    lines.append("To use different models from your collection:")
    lines.append("")
    lines.append(CODE_BLOCK_OPEN)
    lines.append("python scripts/auto_config.py")
    lines.append(CODE_BLOCK_CLOSE)
    lines.append("")
    lines.append("This will scan your Ollama models and let you select which ones to use for your roundtable discussion.")

    md_path = Path("CONFIGURED_MODELS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✓ Generated {md_path}")


if __name__ == "__main__":
    success = update_config()
    sys.exit(0 if success else 1)
