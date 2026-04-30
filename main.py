#!/usr/bin/env python3
"""
LLM Roundtable Discussion Program

A CLI/TUI application for running multi-model discussions with local LLMs via Ollama.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from core import Config
from storage import SessionManager, Exporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Roundtable Discussion Program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  roundtable                          # Launch TUI
  roundtable --prompt "..."           # Start discussion with prompt
  roundtable --prompt-file topic.txt  # Start discussion from file
  roundtable --list-sessions          # List saved sessions
  roundtable --resume SESSION_ID      # Resume a session
  roundtable --export SESSION_ID      # Export a session
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config file (default: ./config.json)",
    )

    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Discussion prompt",
    )

    parser.add_argument(
        "--prompt-file",
        "-f",
        type=str,
        default=None,
        help="File containing discussion prompt",
    )

    parser.add_argument(
        "--tui",
        "-t",
        action="store_true",
        help="Force launch TUI interface",
    )

    parser.add_argument(
        "--list-sessions",
        "-l",
        action="store_true",
        help="List saved discussion sessions",
    )

    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Resume a session by ID",
    )

    parser.add_argument(
        "--export",
        "-e",
        type=str,
        default=None,
        help="Export a session by ID",
    )

    parser.add_argument(
        "--export-format",
        type=str,
        choices=["md", "json"],
        default=None,
        help="Export format (default: from config)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for export",
    )

    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Check if Ollama is available",
    )

    return parser.parse_args()


async def check_ollama(config: Config) -> bool:
    from core import OllamaClient

    client = OllamaClient(
        base_url=config.ollama.base_url,
        timeout=config.ollama.timeout,
    )
    try:
        available = await client.is_available()

        if available:
            models = await client.list_models()
            print(f"✓ Ollama is available at {config.ollama.base_url}")
            print(f"  Found {len(models)} models:")
            for model in models[:10]:
                print(f"      - {model}")
            if len(models) > 10:
                print(f"      ... and {len(models) - 10} more")
        else:
            print(f"✗ Ollama is not available at {config.ollama.base_url}")
            print("  Make sure Ollama is running: ollama serve")

        return available
    finally:
        await client.close()


async def list_sessions(config: Config):
    manager = SessionManager(config.storage.sessions_dir)
    sessions = await manager.list_sessions()

    if not sessions:
        print("No saved sessions found.")
        return

    print(f"Found {len(sessions)} session(s):\n")
    for session in sessions:
        status_icon = "✓" if session["status"] == "completed" else "○"
        if session["status"] == "stopped":
            status_icon = "⏸"
        print(f"{status_icon} {session['id'][:36]}")
        print(f"  Prompt: {session['prompt'][:80]}...")
        print(f"  Status: {session['status']} | Rounds: {session.get('completed_rounds', 0)}")
        print(f"  Created: {session['created_at'][:19]}")
        print()


async def export_session(
    config: Config, session_id: str, output_path: str | None, format: str | None
):
    manager = SessionManager(config.storage.sessions_dir)
    session = await manager.load(session_id)

    if not session:
        print(f"Session {session_id} not found.")
        return

    if format is None:
        format = config.storage.export_format

    if output_path is None:
        output_path = Path(config.storage.sessions_dir) / f"discussion_{session.id[:8]}.{format}"

    output_path = Path(output_path)
    await Exporter.export(session, output_path, format)
    print(f"Exported session to: {output_path}")


async def run_cli_discussion(config: Config, prompt: str):
    from core import DiscussionOrchestrator
    from storage import Session

    print(f"Starting roundtable discussion...")
    print(f"Models: {', '.join(m.name for m in config.models)}")
    print(f"Max rounds: {config.discussion.max_rounds}")
    print(f"Consensus threshold: {config.discussion.consensus_threshold}")
    print()

    session = Session(prompt=prompt, config=config.model_dump())
    orchestrator = DiscussionOrchestrator(
        config=config,
        session=session,
    )

    try:
        result_session = await orchestrator.run()

        print("\n" + "=" * 80)
        print("DISCUSSION COMPLETE")
        print("=" * 80)
        print(f"Status: {result_session.status}")
        print(f"Rounds: {result_session.completed_rounds}")
        print(f"Consensus: {'Yes' if result_session.consensus_reached else 'No'}")
        if result_session.consensus_round:
            print(f"Consensus reached in round: {result_session.consensus_round}")
        print()

        if result_session.summaries:
            print("=== SUMMARIES ===")
            for s in result_session.summaries:
                print(f"\n--- Round {s.round} ---")
                print(s.summary)
            print()

        print("=== DETAILED RESPONSES ===")
        for response in result_session.responses:
            print(f"\n[Round {response.round}] {response.model}:")
            print(response.content)
            print("-" * 40)

        if result_session.similarity_matrices:
            print("\n=== SIMILARITY MATRICES ===")
            for matrix_entry in result_session.similarity_matrices:
                round_num = matrix_entry["round"]
                model_names = matrix_entry["model_names"]
                matrix = matrix_entry["matrix"]
                print(f"\n--- Round {round_num} ---")
                table = Exporter._format_matrix_table(model_names, matrix)
                print(table)
                
                threshold = config.discussion.consensus_threshold
                n = len(model_names)
                agreeing_pairs = sum(
                    1 for i in range(n) for j in range(i + 1, n) if matrix[i][j] >= threshold
                )
                total_pairs = n * (n - 1) // 2 if n > 1 else 0
                agreement_pct = (agreeing_pairs / total_pairs * 100) if total_pairs > 0 else 0
                print(f"\nAgreement: {agreeing_pairs}/{total_pairs} pairs above {threshold} threshold ({agreement_pct:.1f}%)")
            print()
        print()
        print()

        export_path = (
            Path(config.storage.sessions_dir)
            / f"discussion_{session.id[:8]}.{config.storage.export_format}"
        )
        await Exporter.export(result_session, export_path, config.storage.export_format)
        print(f"\nExported to: {export_path}")

    finally:
        await orchestrator.cleanup()


def main():
    args = parse_args()

    try:
        config = Config.load(args.config)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if args.check_ollama:
        available = asyncio.run(check_ollama(config))
        sys.exit(0 if available else 1)

    if args.list_sessions:
        asyncio.run(list_sessions(config))
        return

    if args.resume:
        print("Resume functionality available in TUI mode")
        print("Run: roundtable --tui")
        return

    if args.export:
        asyncio.run(
            export_session(
                config,
                args.export,
                args.output,
                args.export_format,
            )
        )
        return

    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading prompt file: {e}", file=sys.stderr)
            sys.exit(1)

    if prompt and not args.tui:
        asyncio.run(run_cli_discussion(config, prompt))
    else:
        from tui.app import run_tui

        run_tui(config_path=args.config)


if __name__ == "__main__":
    main()
