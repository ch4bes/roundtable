#!/usr/bin/env python3
"""Automatically configure roundtable with your best available Ollama models.

Four levels of configuration:
- Basic: Just select models and moderator (quick start)
- Standard: Common discussion options (recommended)
- Advanced: Context and discussion flow options
- All: Every configuration option
"""

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
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            print("Warning: No models found in 'ollama list' output.")
            return []

        # Parse header to find column positions
        header = lines[0].lower()
        col_names = header.split()
        try:
            name_col = col_names.index("name")
            size_col = col_names.index("size")
        except ValueError:
            # Fallback: assume standard format (name=0, size=2)
            print("Warning: Could not parse 'ollama list' headers. Using fallback column positions.")
            name_col, size_col = 0, 2

        models = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) > max(name_col, size_col):
                name = parts[name_col]
                size_str = parts[size_col]
                size_gb = parse_size(size_str)
                models.append({"name": name, "size_gb": size_gb})
            else:
                print(f"Warning: Skipping unparseable line: {line.strip()}")
        return models
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: Could not get Ollama models: {e}", file=sys.stderr)
        return []
    except (ValueError, IndexError) as e:
        print(f"Error: Failed to parse model list: {e}", file=sys.stderr)
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
    except (ValueError, TypeError):
        return 0


def select_models(all_models: list[dict]) -> list[dict]:
    """Let user manually select which models to include."""
    print("\nAvailable models:")
    for i, model in enumerate(all_models, 1):
        size = f"{model['size_gb']:.1f} GB" if model["size_gb"] > 0 else "cloud"
        print(f"  {i}. {model['name']} ({size})")
    
    print("\nEnter model numbers to include (e.g., '1,3,5' or '1-4' or 'all'):")
    print("Or press Enter to auto-select diverse models")
    user_input = input("Selection: ").strip().lower()
    
    if user_input == "" or user_input == "all":
        print("\nAuto-selecting diverse models...")
        return select_diverse_models(all_models, count=3)
    
    # Parse selection: "1,3,5" or "1-4" or "1 3 5"
    selected = []
    seen_indices = set()
    
    # Handle different separators: comma, space, dash
    for part in user_input.replace(",", " ").split():
        if "-" in part:
            # Handle range like "1-4"
            try:
                start, end = map(int, part.split("-"))
                for idx in range(start, end + 1):
                    if 1 <= idx <= len(all_models) and idx not in seen_indices:
                        selected.append(all_models[idx - 1])
                        seen_indices.add(idx)
            except (ValueError, IndexError):
                pass
        else:
            # Handle single number
            try:
                idx = int(part)
                if 1 <= idx <= len(all_models) and idx not in seen_indices:
                    selected.append(all_models[idx - 1])
                    seen_indices.add(idx)
            except (ValueError, IndexError):
                pass
    
    if not selected:
        print("\nNo valid selection, auto-selecting diverse models...")
        return select_diverse_models(all_models, count=3)
    
    return selected


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


def prompt_yes_no(prompt_text: str, default: bool) -> bool:
    """Prompt user with a yes/no question, returning boolean."""
    default_str = "Y" if default else "N"
    user_input = input(f"{prompt_text} (Y/n, default: {default_str}): ").strip().lower()
    if user_input == "":
        return default
    if user_input in ["y", "yes"]:
        return True
    if user_input in ["n", "no"]:
        return False
    print(f"  Invalid input, using default: {default_str}")
    return default


def select_from_options(prompt_text: str, options: list[tuple[str, str]], default_idx: int = 0) -> str:
    """Prompt user to select from a list of options."""
    print(f"\n{prompt_text}")
    for i, (value, description) in enumerate(options, 1):
        marker = " (default)" if i - 1 == default_idx else ""
        print(f"  {i}. {value} - {description}{marker}")
    
    while True:
        user_input = input(f"Selection (1-{len(options)}): ").strip()
        if user_input == "":
            return options[default_idx][0]
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
            else:
                print(f"  Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("  Please enter a number from the list.")


def select_moderator(all_models: list[dict]) -> dict:
    """Let user select a moderator from all available Ollama models."""
    print("\nAll available models for moderator selection:")
    for i, model in enumerate(all_models, 1):
        size = f"{model['size_gb']:.1f} GB" if model["size_gb"] > 0 else "cloud"
        print(f"  {i}. {model['name']} ({size})")
    
    while True:
        user_input = input(f"\nWhich model should be moderator? (1-{len(all_models)}): ").strip()
        if user_input == "":
            print(f"  Defaulting to: {all_models[0]['name']}")
            return all_models[0]
        try:
            choice = int(user_input)
            if 1 <= choice <= len(all_models):
                return all_models[choice - 1]
            else:
                print(f"  Please enter a number between 1 and {len(all_models)}")
        except ValueError:
            print("  Please enter a number from the list.")


def configure_basic(all_models: list[dict]) -> dict:
    """Basic configuration - model selection."""
    print("\n" + "=" * 50)
    print("STEP 1: SELECT MODELS")
    print("=" * 50)
    
    # Select participants
    selected = select_models(all_models)
    
    # Select moderator
    moderator = select_moderator(all_models)
    
    return {
        "models": selected,
        "moderator": moderator,
    }


def configure_standard(all_models: list[dict], basic_result: dict) -> dict:
    """Standard configuration - discussion settings."""
    print("\n" + "=" * 50)
    print("STEP 2: DISCUSSION SETTINGS")
    print("=" * 50)
    
    max_rounds = prompt_with_default("Max discussion rounds?", 6)
    consensus_threshold = prompt_float_with_default("Consensus threshold (0-1)?", 0.75)
    human_enabled = prompt_yes_no("Allow human participation?", True)
    
    return {
        "max_rounds": max_rounds,
        "consensus_threshold": consensus_threshold,
        "human_enabled": human_enabled,
    }


def configure_advanced(standard_result: dict) -> dict:
    """Advanced configuration - context mode and discussion flow."""
    print("\n" + "=" * 50)
    print("STEP 3: CONTEXT & FLOW")
    print("=" * 50)
    
    # Context mode
    context_mode = select_from_options(
        "What should models see when responding?",
        [
            ("summary_only", "Moderator summary only (recommended)"),
            ("summary_plus_last_n", "Summary + last N responses"),
            ("full", "Full discussion history"),
        ],
        default_idx=0
    )
    
    last_n = 2
    preview_length = 800  # default
    if context_mode == "summary_plus_last_n":
        last_n = prompt_with_default("How many recent responses to include?", 2)
        show_full = prompt_yes_no("Show full responses (no truncation)?", False)
        if show_full:
            preview_length = 0  # 0 means show all
        else:
            preview_length = prompt_with_default("Characters to show per response?", 800)
    
    # Discussion flow
    rotation = select_from_options(
        "Model response order?",
        [
            ("sequential", "Rotate through list each round"),
            ("random", "Shuffle order each round"),
            ("fixed", "Same order every round"),
        ],
        default_idx=0
    )
    
    final_review = prompt_yes_no("Generate final review when consensus reached?", True)
    
    return {
        "context_mode": context_mode,
        "last_n": last_n,
        "preview_length": preview_length if context_mode == "summary_plus_last_n" else 800,
        "rotation": rotation,
        "final_review": final_review,
    }


def configure_all(advanced_result: dict) -> dict:
    """All options - model params, consensus, and storage."""
    print("\n" + "=" * 50)
    print("STEP 4: ADVANCED OPTIONS")
    print("=" * 50)
    
    # Model parameters
    print("\n--- Model Parameters ---")
    temperature = prompt_float_with_default("Participant temperature (0-2)?", 0.7)
    max_tokens = prompt_with_default("Max tokens per response?", 2048)
    context_window = prompt_with_default("Context window size?", 32768)
    mod_temp = prompt_float_with_default("Moderator temperature?", 0.5)
    
    # Consensus settings
    print("\n--- Consensus ---")
    consensus_mode = select_from_options(
        "How should consensus be determined?",
        [
            ("moderator_decides", "LLM moderator analyzes and decides"),
            ("programmatic_decides", "Program calculates from similarity"),
        ],
        default_idx=0
    )
    
    consensus_method = select_from_options(
        "Similarity method?",
        [
            ("clustering", "Clustering-based (flexible)"),
            ("pairwise", "Pairwise all-or-nothing (strict)"),
        ],
        default_idx=0
    )
    
    strictness = select_from_options(
        "Consensus strictness?",
        [
            ("main_point", "Main point agreement (lenient)"),
            ("full", "Full agreement required (strict)"),
        ],
        default_idx=0
    )
    
    # Only ask about thresholds if clustering
    if consensus_method == "clustering":
        agree_reached = prompt_float_with_default(
            "Similarity threshold when moderator says REACHED?", 0.50
        )
        agree_not_reached = prompt_float_with_default(
            "Similarity threshold when moderator says NOT REACHED?", 0.75
        )
        reprompt_pct = prompt_float_with_default(
            "Agreement % to reprompt moderator?", 0.70
        )
    else:
        agree_reached = 0.50
        agree_not_reached = 0.75
        reprompt_pct = 0.70
    
    # Storage
    print("\n--- Storage ---")
    sessions_dir = input("Sessions directory? (default: ./sessions): ").strip() or "./sessions"
    auto_save = prompt_yes_no("Auto-save sessions?", True)
    export_fmt = select_from_options(
        "Export format?",
        [
            ("md", "Markdown"),
            ("json", "JSON"),
        ],
        default_idx=0
    )
    
    # Human customization
    print("\n--- Human Participant ---")
    human_prompt = input(
        "Human prompt template? (default: 'Share your perspective on: {prompt}'): "
    ).strip() or "Share your perspective on: {prompt}"
    human_name = input("Human display name? (default: Human): ").strip() or "Human"
    
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_ctx": context_window,
        "moderator_temp": mod_temp,
        "consensus_mode": consensus_mode,
        "consensus_method": consensus_method,
        "strictness": strictness,
        "agree_reached": agree_reached,
        "agree_not_reached": agree_not_reached,
        "reprompt_pct": reprompt_pct,
        "sessions_dir": sessions_dir,
        "auto_save": auto_save,
        "export_fmt": export_fmt,
        "human_prompt": human_prompt,
        "human_name": human_name,
    }


def update_config():
    """Auto-configure roundtable with guided setup."""
    print("\n" + "=" * 60)
    print("  ROUNDTABLE CONFIGURATION WIZARD")
    print("=" * 60)
    print("\nI'll help you set up your roundtable discussion.")
    print("\nChoose your configuration level:")
    print("  1. Basic - Select models and moderator")
    print("  2. Standard - Basic + rounds, consensus, human")
    print("  3. Advanced - Standard + context mode, flow")
    print("  4. All - Everything (all options)")
    
    while True:
        level = input("\nChoose level (1-4): ").strip()
        if level in ["1", "2", "3", "4"]:
            break
        print("  Please enter 1, 2, 3, or 4")
    
    # Scan models
    print("\nScanning Ollama models...")
    all_models = get_ollama_models()
    if not all_models:
        print("✗ No models found. Make sure Ollama is running.")
        return False
    
    print(f"✓ Found {len(all_models)} models\n")
    
    # Step 1: Basic (always required)
    basic_result = configure_basic(all_models)
    
    # Step 2: Standard
    if level in ["2", "3", "4"]:
        standard_result = configure_standard(all_models, basic_result)
    else:
        standard_result = {
            "max_rounds": 6,
            "consensus_threshold": 0.75,
            "human_enabled": True,
        }
    
    # Step 3: Advanced
    if level in ["3", "4"]:
        advanced_result = configure_advanced(standard_result)
    else:
        advanced_result = {
            "context_mode": "summary_only",
            "last_n": 2,
            "preview_length": 800,
            "rotation": "sequential",
            "final_review": True,
        }
    
    # Step 4: All
    if level == "4":
        all_result = configure_all(advanced_result)
    else:
        all_result = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "num_ctx": 32768,
            "moderator_temp": 0.5,
            "consensus_mode": "moderator_decides",
            "consensus_method": "clustering",
            "strictness": "main_point",
            "agree_reached": 0.50,
            "agree_not_reached": 0.75,
            "reprompt_pct": 0.70,
            "sessions_dir": "./sessions",
            "auto_save": True,
            "export_fmt": "md",
            "human_prompt": "Share your perspective on: {prompt}",
            "human_name": "Human",
        }
    
    # Build config
    selected = basic_result["models"]
    moderator = basic_result["moderator"]
    embedding_model = has_embedding_model(all_models)
    
    config = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "timeout": 300
        },
        "models": [
            {
                "name": m["name"],
                "temperature": all_result["temperature"],
                "max_tokens": all_result["max_tokens"],
                "num_ctx": all_result["num_ctx"],
            }
            for m in selected
        ],
        "moderator": {
            "name": moderator["name"],
            "temperature": all_result["moderator_temp"],
            "max_tokens": all_result["max_tokens"],
            "num_ctx": all_result["num_ctx"],
        },
        "discussion": {
            "max_rounds": standard_result["max_rounds"],
            "consensus_threshold": standard_result["consensus_threshold"],
            "consensus_method": all_result["consensus_method"],
            "rotation_order": advanced_result["rotation"],
            "final_review_enabled": advanced_result["final_review"],
            "consensus_agreement_when_reached": all_result["agree_reached"],
            "consensus_agreement_when_not_reached": all_result["agree_not_reached"],
            "reprompt_agreement_threshold": all_result["reprompt_pct"],
        },
        "context": {
            "mode": advanced_result["context_mode"],
            "last_n_responses": advanced_result["last_n"],
            "response_preview_length": advanced_result["preview_length"],
        },
        "storage": {
            "sessions_dir": all_result["sessions_dir"],
            "auto_save": all_result["auto_save"],
            "export_format": all_result["export_fmt"],
        },
        "default_prompt": "",
        "embeddings": {
            "model": embedding_model if embedding_model else "qwen3-embedding:8b"
        },
        "human_participant": {
            "enabled": standard_result["human_enabled"],
            "prompt": all_result["human_prompt"],
            "display_name": all_result["human_name"],
        },
        "consensus": {
            "mode": all_result["consensus_mode"],
            "threshold": standard_result["consensus_threshold"],
            "method": all_result["consensus_method"],
            "strictness": all_result["strictness"],
        },
    }
    
    # Summary
    level_names = {"1": "Basic", "2": "Standard", "3": "Advanced", "4": "All"}
    print("\n" + "=" * 50)
    print(f"CONFIGURATION COMPLETE ({level_names[level]})")
    print("=" * 50)
    print(f"\nParticipants ({len(selected)}):")
    for model in selected:
        size = f"{model['size_gb']:.1f} GB" if model["size_gb"] > 0 else "cloud"
        print(f"  - {model['name']} ({size})")
    
    print(f"\nModerator: {moderator['name']}")
    print(f"Max rounds: {standard_result['max_rounds']}")
    print(f"Consensus: {standard_result['consensus_threshold']} ({all_result['consensus_method']})")
    print(f"Context: {advanced_result['context_mode'].replace('_', ' ')}")
    print(f"Human: {'Yes' if standard_result['human_enabled'] else 'No'}")
    
    if embedding_model:
        print(f"Embedding: {embedding_model}")
    else:
        print("\n⚠ No embedding model - using fallback similarity")
    
    # Write files
    config_path = Path("config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    generate_configured_models_md(
        selected, moderator,
        all_result["moderator_temp"],
        embedding_model,
        standard_result["max_rounds"],
        standard_result["consensus_threshold"],
        advanced_result["context_mode"],
        all_result["strictness"],
        level_names[level]
    )
    
    print(f"\n✓ Saved to {config_path}")
    print("\nReady! Run: roundtable")
    
    return True


def generate_configured_models_md(
    selected: list[dict], 
    moderator: dict,
    moderator_temp: float,
    embedding_model: str | None, 
    max_rounds: int,
    consensus_threshold: float,
    context_mode: str,
    strictness: str,
    level: str,
) -> None:
    """Generate CONFIGURED_MODELS.md based on current configuration."""
    moderator_name = moderator["name"]

    lines = []
    lines.append("# Your Configured Models")
    lines.append("")
    lines.append("## Current Configuration")
    lines.append("")

    model_count = len(selected)
    lines.append(f"Your roundtable is configured with **{model_count} models** ({level} setup):\n")

    lines.append("| Model | Size | Role |")
    lines.append("|-------|------|------|")
    for model in selected:
        size = f"{model['size_gb']:.1f} GB" if model['size_gb'] > 0 else "Cloud"
        role = "Moderator" if model["name"] == moderator_name else "Participant"
        lines.append(f"| `{model['name']}` | {size} | {role} |")

    lines.append("")
    lines.append("## Configuration Details")
    lines.append("")
    lines.append(f"- **Moderator**: {moderator_name} (temperature: {moderator_temp})")
    lines.append(f"- **Max Rounds**: {max_rounds}")
    lines.append(f"- **Consensus Threshold**: {consensus_threshold} ({int(consensus_threshold * 100)}% similarity)")
    lines.append(f"- **Context Mode**: {context_mode.replace('_', ' ').title()}")
    lines.append(f"- **Consensus Strictness**: {'Main point (lenient)' if strictness == 'main_point' else 'Full agreement (strict)'}")
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
    lines.append("To adjust your configuration:")
    lines.append("")
    lines.append(CODE_BLOCK_OPEN)
    lines.append("python scripts/auto_config.py")
    lines.append(CODE_BLOCK_CLOSE)
    lines.append("")

    md_path = Path("CONFIGURED_MODELS.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✓ Generated {md_path}")


if __name__ == "__main__":
    success = update_config()
    sys.exit(0 if success else 1)