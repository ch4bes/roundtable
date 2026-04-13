#!/usr/bin/env python3
"""
Automatically configure roundtable with your best available Ollama models.
Selects models of different sizes for diverse perspectives.
"""

import json
import subprocess
import sys
from pathlib import Path


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


def update_config():
    """Auto-configure roundtable with best available models."""
    print("Scanning Ollama models...")
    all_models = get_ollama_models()

    if not all_models:
        print("✗ No models found. Make sure Ollama is running.")
        return False

    print(f"✓ Found {len(all_models)} models\n")

    selected = select_diverse_models(all_models, count=3)
    embedding_model = has_embedding_model(all_models)

    print("Selected models for roundtable:")
    for i, model in enumerate(selected, 1):
        size = f"{model['size_gb']:.1f} GB" if model["size_gb"] > 0 else "cloud"
        print(f"  {i}. {model['name']} ({size})")

    if embedding_model:
        print(f"\n✓ Embedding model found: {embedding_model}")
    else:
        print("\n⚠ No embedding model found")
        print("  Install with: ollama pull nomic-embed-text")
        print("  (Will use fallback text-based similarity)")

    config = {
        "ollama": {"base_url": "http://localhost:11434", "timeout": 120},
        "models": [{"name": m["name"], "temperature": 0.7, "max_tokens": 1024} for m in selected],
        "moderator": {
            "name": selected[len(selected) // 2]["name"],
            "temperature": 0.5,
            "max_tokens": 512,
        },
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

    if embedding_model:
        config["embeddings"] = {"model": embedding_model}
    else:
        config["embeddings"] = {"model": "nomic-embed-text"}

    config_path = Path("config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Updated {config_path}")
    print("\nReady to start! Run: roundtable")

    return True


if __name__ == "__main__":
    success = update_config()
    sys.exit(0 if success else 1)
