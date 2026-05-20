"""
Security utilities for path sanitization and validation.

Prevents path traversal attacks by ensuring all file operations remain
within their intended directories.
"""

import os
import re
from pathlib import Path


# Characters that are dangerous in session IDs / filenames
# Allows: alphanumeric, hyphens, underscores, dots, colons (for Ollama model names), slashes only for subdirs
_SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-./:]+$')

# Characters that are safe for export filenames
_EXPORT_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\\.]+$')


def sanitize_session_id(session_id: str) -> str:
    """Sanitize a session ID to prevent path traversal.

    Strips any directory traversal sequences and ensures only safe
    alphanumeric characters remain. Returns the sanitized string.

    Args:
        session_id: Raw session identifier from user input.

    Returns:
        Sanitized session ID safe for use in file paths.

    Raises:
        ValueError: If the session ID contains disallowed characters.
    """
    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty")

    # Reject path traversal markers explicitly
    if '..' in session_id:
        raise ValueError("Session ID contains invalid path traversal sequence")

    # Extract only the filename component (strips any directory prefixes)
    clean = os.path.basename(session_id.strip())

    # Reject if after basename extraction the result is empty
    if not clean:
        raise ValueError("Session ID is invalid")

    # Reject characters that are dangerous in filenames
    if not _SESSION_ID_PATTERN.match(clean):
        raise ValueError(
            f"Session ID contains disallowed characters: {clean!r}"
        )

    return clean


def sanitize_export_path(output_path: Path, sessions_dir: Path) -> Path:
    """Sanitize and validate an export output path.

    Strips directory traversal components and ensures the resolved path
    stays within the sessions directory.

    Args:
        output_path: Raw output path from user input.
        sessions_dir: The permitted parent directory (from config).

    Returns:
        Sanitized Path safe for export.

    Raises:
        ValueError: If the path would escape the sessions directory.
    """
    if not output_path or not str(output_path).strip():
        raise ValueError("Export path cannot be empty")

    if not sessions_dir:
        raise ValueError("Sessions directory cannot be empty")

    # Extract only the filename -- strips any directory traversal
    filename = os.path.basename(str(output_path))

    # Validate filename contains only safe characters
    if not filename or not _EXPORT_FILENAME_PATTERN.match(filename):
        raise ValueError(
            f"Export filename contains disallowed characters: {filename!r}"
        )

    # Build path under sessions_dir, then resolve to catch symlink escapes
    target_dir = sessions_dir.resolve()
    sanitized_path = target_dir / filename
    resolved_path = sanitized_path.resolve()

    # Validate the resolved path is within the intended directory
    # (catches symlink escapes and other edge cases)
    try:
        resolved_path.relative_to(target_dir)
    except ValueError:
        raise ValueError(
            f"Export path escapes allowed directory: {output_path}"
        )

    return resolved_path


def validate_path_within(parent: Path, child: Path) -> Path:
    """Validate that a resolved child path is within a parent directory.

    This is a general-purpose guard used after path construction to
    catch edge cases that simple string sanitization might miss
    (e.g., symlink escapes).

    Args:
        parent: The permitted parent directory (must be absolute/resolved).
        child: The resolved child path to validate.

    Returns:
        The validated child path.

    Raises:
        ValueError: If child would escape parent directory.
    """
    try:
        child.relative_to(parent)
    except ValueError:
        raise ValueError(
            f"Path escapes allowed directory: {child} is not under {parent}"
        )
    return child
