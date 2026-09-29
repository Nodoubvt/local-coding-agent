"""File-system tools exposed to the model."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from qai.tools.base import ToolResult
from qai.workspace import SandboxError, Workspace

MAX_READ_BYTES = 200_000
MAX_READ_LINES = 2_000
MAX_DIFF_CHARS = 4_000
MAX_GREP_HITS = 100


def _text(path: Path, limit: int = MAX_READ_BYTES) -> str:
    raw = path.read_bytes()[: limit + 1]
    if b"\x00" in raw[:4096]:
        raise ValueError("file looks binary; use grep or read_file with a line range")
    return raw.decode("utf-8", errors="replace")


def _read_file(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    path = workspace.resolve(args.get("path", ""))
    if not path.exists():
        return ToolResult.error(f"file not found: {workspace.relative(path)}")
    if path.is_dir():
        return ToolResult.error(
            f"{workspace.relative(path)} is a directory; use list_dir"
        )
    if workspace.is_ignored(path):
        return ToolResult.error(f"{workspace.relative(path)} is excluded by ignore rules")

    content = _text(path)
    lines = content.splitlines()
    start = max(1, int(args.get("start_line") or 1))
    end = int(args.get("end_line") or 0) or len(lines)
    end = min(end, len(lines))
    if start > len(lines):
        return ToolResult.error(f"start_line {start} is past end of file ({len(lines)} lines)")

    selected = lines[start - 1 : end]
    if args.get("numbered"):
        width = len(str(end))
        body = "\n".join(f"{i:>{width}}\t{line}" for i, line in enumerate(selected, start))
    else:
        body = "\n".join(selected)

    header = f"{workspace.relative(path)} (lines {start}-{end} of {len(lines)})"
    return ToolResult(content=f"{header}\n{body}")


def _write_file(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    path = workspace.resolve(args.get("path", ""))
    content = args.get("content")
    if not isinstance(content, str):
        return ToolResult.error("write_file requires a 'content' string")

    existed = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")

    verb = "updated" if existed else "created"
    lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    return ToolResult(content=f"{verb} {workspace.relative(path)} ({lines} lines)")


def _edit_file(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    path = workspace.resolve(args.get("path", ""))
    old = args.get("old_string")
    new = args.get("new_string", "")
    if not isinstance(old, str) or not old:
        return ToolResult.error("edit_file requires a non-empty 'old_string'")
    if not isinstance(new, str):
        return ToolResult.error("edit_file requires 'new_string' to be a string")
    if not path.is_file():
        return ToolResult.error(f"file not found: {workspace.relative(path)}")

    content = _text(path)
    occurrences = content.count(old)
    if occurrences == 0:
        return ToolResult.error(
            f"old_string not found in {workspace.relative(path)}; read the file again"
        )
    if occurrences > 1 and not args.get("replace_all"):
        return ToolResult.error(
            f"old_string appears {occurrences} times in {workspace.relative(path)}; "
            "add more surrounding context or pass replace_all"
        )

    updated = content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1)
    path.write_text(updated, encoding="utf-8", newline="\n")
    count = occurrences if args.get("replace_all") else 1
    return ToolResult(
        content=f"edited {workspace.relative(path)} ({count} replacement{'s' if count > 1 else ''})"
    )


def _list_dir(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    path = workspace.resolve(args.get("path") or ".")
    if not path.exists():
        return ToolResult.error(f"directory not found: {workspace.relative(path)}")
    if not path.is_dir():
        return ToolResult.error(f"{workspace.relative(path)} is not a directory")

    depth = max(1, min(int(args.get("depth") or 1), 5))
    entries: list[str] = []
    for child in sorted(path.rglob("*")):
        if len(child.relative_to(path).parts) > depth:
            continue
        if workspace.is_ignored(child) or any(
            workspace.is_ignored(parent) for parent in child.parents if parent != workspace.root
        ):
            continue
        suffix = "/" if child.is_dir() else ""
        entries.append(child.relative_to(path).as_posix() + suffix)

    if not entries:
        return ToolResult(content=f"{workspace.relative(path)} is empty")
    body = "\n".join(entries[:500])
    if len(entries) > 500:
        body += f"\n... {len(entries) - 500} more"
    return ToolResult(content=f"{workspace.relative(path)}/\n{body}")


def _grep(args: dict[str, Any], workspace: Workspace) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return ToolResult.error("grep requires a 'pattern' string")
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult.error(f"invalid regular expression: {exc}")

    root = workspace.resolve(args.get("path") or ".")
    if not root.exists():
        return ToolResult.error(f"path not found: {workspace.relative(root)}")

    glob = args.get("glob") or ""
    hits: list[str] = []
    files = [root] if root.is_file() else workspace.walk(root)
    for file in files:
        name = file.name
        rel = file.as_posix()
        if glob and not (Path(name).match(glob) or rel.endswith(glob) or rel == glob):
            continue
        try:
            for number, line in enumerate(_text(file).splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{workspace.relative(file)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= MAX_GREP_HITS:
                        break
        except (OSError, ValueError):
            continue
        if len(hits) >= MAX_GREP_HITS:
            break

    if not hits:
        return ToolResult(content=f"no matches for {pattern!r}")
    suffix = "\n... (truncated)" if len(hits) >= MAX_GREP_HITS else ""
    return ToolResult(content="\n".join(hits) + suffix)


def _diff(path: Workspace, args: dict[str, Any], limit: int = MAX_DIFF_CHARS) -> str:
    target = path.resolve(args.get("path", ""))
    before = _text(target) if target.is_file() else ""
    after = args.get("content", "")
    if not isinstance(after, str):
        return "(no preview)"
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path.relative(target)}",
            tofile=f"b/{path.relative(target)}",
            n=3,
        )
    )
    if not lines:
        return "(no changes)"
    body = "".join(lines)
    return body if len(body) <= limit else body[:limit] + "\n... (diff truncated)"


READ_FILE = {
    "name": "read_file",
    "description": (
        "Read a UTF-8 text file inside the workspace. "
        "Use numbered=true to get line numbers."
    ),
    "mutating": False,
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative path."},
            "start_line": {"type": "integer", "description": "1-based first line (default 1)."},
            "end_line": {"type": "integer", "description": "1-based last line, 0 or omit for EOF."},
            "numbered": {"type": "boolean", "description": "Prefix each line with its number."},
        },
        "required": ["path"],
    },
}

WRITE_FILE = {
    "name": "write_file",
    "description": "Create a new file or replace a whole file's contents. Requires user approval.",
    "mutating": True,
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative path."},
            "content": {"type": "string", "description": "Full new file contents."},
        },
        "required": ["path", "content"],
    },
}

EDIT_FILE = {
    "name": "edit_file",
    "description": (
        "Replace an exact snippet in an existing file. old_string must be unique "
        "unless replace_all is true. Requires user approval."
    ),
    "mutating": True,
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
        },
        "required": ["path", "old_string", "new_string"],
    },
}

LIST_DIR = {
    "name": "list_dir",
    "description": "List files and directories in the workspace. Ignored paths are hidden.",
    "mutating": False,
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list, default '.'"},
            "depth": {"type": "integer", "description": "Recursion depth 1-5 (default 1)."},
        },
    },
}

GREP = {
    "name": "grep",
    "description": "Search the workspace for a regular expression. Returns path:line matches.",
    "mutating": False,
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "File or directory to search (default '.')."},
            "glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'."},
        },
        "required": ["pattern"],
    },
}


class _CallableTool:
    """Adapts a plain function + schema dict into a :class:`Tool`."""

    def __init__(self, schema: dict[str, Any], func: Any) -> None:
        self.name: str = schema["name"]
        self.description: str = schema["description"]
        self.mutating: bool = schema["mutating"]
        self.parameters: dict[str, Any] = schema["parameters"]
        self._func = func

    def run(self, args: dict[str, Any], workspace: Workspace) -> ToolResult:
        try:
            return self._func(args, workspace)
        except SandboxError as exc:
            return ToolResult.error(str(exc))
        except FileNotFoundError as exc:
            return ToolResult.error(f"not found: {exc}")
        except IsADirectoryError as exc:
            return ToolResult.error(f"is a directory: {exc}")
        except OSError as exc:
            return ToolResult.error(f"filesystem error: {exc}")

    def describe(self, args: dict[str, Any], workspace: Workspace) -> tuple[str, str | None]:
        if self.name == "write_file":
            return f"write {workspace.relative(workspace.resolve(args.get('path', '')))}", _diff(
                workspace, args
            )
        if self.name == "edit_file":
            return f"edit {workspace.relative(workspace.resolve(args.get('path', '')))}", None
        return f"{self.name} {args.get('path', '')}".strip(), None


def default_tools() -> list[_CallableTool]:
    return [
        _CallableTool(READ_FILE, _read_file),
        _CallableTool(EDIT_FILE, _edit_file),
        _CallableTool(WRITE_FILE, _write_file),
        _CallableTool(LIST_DIR, _list_dir),
        _CallableTool(GREP, _grep),
    ]
