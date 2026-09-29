"""Context assembly: @file mentions, --attach paths and a workspace manifest."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from qai.workspace import SandboxError, Workspace

MENTION_RE = re.compile(r"(?<![\w@])@([\w./\\-]+)")
MAX_ATTACH_BYTES = 24_000
MAX_TREE_ENTRIES = 120

TRUNCATION_NOTE = "\n... [truncated]"


def extract_mentions(text: str) -> tuple[str, list[str]]:
    """Pull ``@path`` mentions out of *text*.

    Returns the text with mentions removed and the list of referenced paths.
    """
    mentions: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1).rstrip(".,;:")
        mentions.append(token)
        return ""

    return MENTION_RE.sub(_replace, text), mentions


def expand(workspace: Workspace, spec: str) -> list[Path]:
    """Expand a path or glob pattern into a list of files inside the workspace."""
    if any(ch in spec for ch in "*?["):
        return _expand_glob(workspace, spec)

    try:
        resolved = workspace.resolve(spec)
    except SandboxError:
        return []
    if resolved.is_dir():
        return [p for p in workspace.walk(resolved) if not workspace.is_ignored(p)]
    if resolved.is_file() and not workspace.is_ignored(resolved):
        return [resolved]
    return []


def _expand_glob(workspace: Workspace, spec: str) -> list[Path]:
    """Glob relative to the first literal path segment, e.g. ``src/**/*.py``."""
    literal: list[str] = []
    for part in Path(spec).parts:
        if any(ch in part for ch in "*?["):
            break
        literal.append(part)
    else:
        literal = []

    if literal:
        prefix = Path(*literal)
        base = workspace.resolve(prefix) if prefix.parts else workspace.root
        pattern = spec[len(str(prefix).replace("\\", "/")) :].lstrip("/")
    else:
        base = workspace.root
        pattern = spec

    if not base.is_dir():
        return []

    try:
        matches = sorted(p for p in base.glob(pattern) if p.is_file())
    except (OSError, ValueError):
        return []

    out: list[Path] = []
    for match in matches:
        try:
            resolved = workspace.resolve(match)
        except SandboxError:
            continue
        if not workspace.is_ignored(resolved):
            out.append(resolved)
    return out


def _render_file(path: Path, workspace: Workspace) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    try:
        content = raw[:MAX_ATTACH_BYTES].decode("utf-8", errors="replace")
    except Exception:
        return None
    if len(raw) > MAX_ATTACH_BYTES:
        content += TRUNCATION_NOTE
    return f'<file path="{workspace.relative(path)}">\n{content}\n</file>'


def build_manifest(workspace: Workspace, max_entries: int = MAX_TREE_ENTRIES) -> str:
    """A compact listing of the workspace so the model knows what exists."""
    entries: list[str] = []
    for path in workspace.walk(max_files=max_entries * 4):
        entries.append(workspace.relative(path))
        if len(entries) >= max_entries:
            break
    if not entries:
        return "<workspace>\n(empty)\n</workspace>"
    body = "\n".join(entries)
    more = "\n... (truncated)" if len(entries) >= max_entries else ""
    return f"<workspace root=\"{workspace.root}\">\n{body}{more}\n</workspace>"


def collect_context(
    workspace: Workspace, mentions: list[str], include_manifest: bool = True
) -> str:
    """Build the system-adjacent context block for the given mentions."""
    chunks: list[str] = []
    if include_manifest:
        chunks.append(build_manifest(workspace))

    seen: set[str] = set()
    for spec in mentions:
        for path in expand(workspace, spec):
            rel = workspace.relative(path)
            if rel in seen:
                continue
            seen.add(rel)
            rendered = _render_file(path, workspace)
            if rendered is not None:
                chunks.append(rendered)

    return "\n\n".join(chunks) if chunks else ""


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
