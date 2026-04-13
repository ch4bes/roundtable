#!/usr/bin/env python3
"""
Utility to update config.json with available Ollama models.
"""

import json
import subprocess
import sys
from pathlib import Path


def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")[1:]  # Skip header
        models = []
        for line in lines:
            if line.strip():
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama list: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("Error: ollama not found. Make sure Ollama is installed.", file=sys.stderr)
        return []


def update_config(config_path: str = "config.json", num_models: int = 3):
    """Update config.json with available models."""

    models = get_ollama_models()

    if not models:
        print("No models found. Make sure Ollama is running and has models installed.")
        return False

    print(f"Found {len(models)} Ollama models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()

    # Load existing config or create new
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        config = {
            "ollama": {"base_url": "http://localhost:11434", "timeout": 120},
            "discussion": {
                "max_rounds": 10,
                "consensus_threshold": 0.85,
                "rotation_order": "sequential",
            },
            "context": {"mode": "summary_only", "last_n_responses": 2},
            "storage": {
                "sessions_dir": "./sessions",
                "auto_save": True,
                "export_format": "md",
            },
            "default_prompt": "",
        }

    # Select models (prefer different sizes for diversity)
    selected_models = models[:num_models] if len(models) >= num_models else models

    if len(selected_models) < 2:
        print(f"Error: Need at least 2 models for roundtable, found {len(selected_models)}")
        return False

    # Update config
    config["models"] = [
        {"name": m, "temperature": 0.7, "max_tokens": 1024} for m in selected_models
    ]

    # Set moderator to middle-sized model or first model
    moderator_idx = len(selected_models) // 2
    config["moderator"] = {
        "name": selected_models[moderator_idx],
        "temperature": 0.5,
        "max_tokens": 512,
    }

    # Set embeddings model (check if available)
    embedding_models = [m for m in models if "embed" in m.lower()]
    if embedding_models:
        config["embeddings"] = {"model": embedding_models[0]}
        print(f"Using embedding model: {embedding_models[0]}")
    else:
        print("Warning: No embedding model found. Install one with: ollama pull nomic-embed-text")
        config["embeddings"] = {"model": "nomic-embed-text"}

    # Save config
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Updated {config_path}")
    print(f"\nSelected models:")
    for model in config["models"]:
        print(f"  - {model['name']}")
    print(f"\nModerator: {config['moderator']['name']}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update roundtable config with Ollama models")
    parser.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Path to config file (default: config.json)",
    )
    parser.add_argument(
        "--num-models",
        "-n",
        type=int,
        default=3,
        help="Number of models to select (default: 3)",
    )

    args = parser.parse_args()

    success = update_config(args.config, args.num_models)
    sys.exit(0 if success else 1)
