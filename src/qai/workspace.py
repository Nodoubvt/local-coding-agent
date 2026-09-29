"""Workspace root resolution, path sandboxing and ignore rules."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

import pathspec

DEFAULT_IGNORES = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "*.pyc",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.gguf",
    "*.safetensors",
    "*.bin",
)

TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".jsonc",
    ".md", ".markdown", ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".c", ".h", ".cc", ".cpp",
    ".hpp", ".cs", ".java", ".kt", ".kts", ".go", ".rs", ".rb", ".php", ".swift",
    ".sql", ".html", ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".xml",
    ".gradle", ".lock", ".env", ".gitignore", ".editorconfig", "",
}
TEXT_NAMES = {"dockerfile", "makefile", "readme", "license", "changelog", "justfile"}

# pathspec renamed the gitignore factory in 1.0; support both spellings.
_IGNORE_FACTORY = "gitignore" if hasattr(pathspec, "GitIgnoreSpec") else "gitwildmatch"


class SandboxError(PermissionError):
    """Raised when a path escapes the workspace or is explicitly ignored."""


class Workspace:
    """A sandboxed directory tree that tools are allowed to touch."""

    def __init__(self, root: Path, extra_ignores: list[str] | None = None) -> None:
        self.root = Path(os.path.expanduser(str(root))).resolve()
        self._ignores = pathspec.PathSpec.from_lines(
            _IGNORE_FACTORY, [*DEFAULT_IGNORES, *(extra_ignores or [])]
        )
        self.user_ignores: list[str] = []

    # -- path handling ---------------------------------------------------
    def resolve(self, raw: str | Path) -> Path:
        """Resolve *raw* against the workspace root, refusing to escape it."""
        text = str(raw).strip().strip('"').strip("'")
        if not text:
            raise SandboxError("empty path")
        if "\x00" in text:
            raise SandboxError("path contains a null byte")

        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate

        resolved = _resolve_lenient(candidate)
        if resolved != self.root and self.root not in resolved.parents:
            raise SandboxError(
                f"path escapes the workspace root ({self.root}): {raw}"
            )
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix() or "."
        except ValueError:
            return path.as_posix()

    def is_ignored(self, path: Path) -> bool:
        rel = self.relative(path)
        if rel in {".", ""}:
            return False
        if self._ignores.match_file(rel) or self._ignores.match_file(rel + "/"):
            return True
        return any(fnmatch.fnmatch(rel, pat) for pat in self.user_ignores)

    def is_text_file(self, path: Path) -> bool:
        if path.suffix.lower() in TEXT_SUFFIXES:
            return True
        return path.name.lower() in TEXT_NAMES

    # -- traversal -------------------------------------------------------
    def walk(self, start: Path | None = None, max_files: int = 2000) -> list[Path]:
        base = self.resolve(start) if start is not None else self.root
        if base.is_file():
            return [base]
        found: list[Path] = []
        for path in sorted(base.rglob("*")):
            if len(found) >= max_files:
                break
            if path.is_dir() or self.is_ignored(path) or not self.is_text_file(path):
                continue
            found.append(path)
        return found

    def add_ignores(self, patterns: list[str]) -> None:
        self.user_ignores.extend(patterns)


def _resolve_lenient(path: Path) -> Path:
    """Resolve symlinks as far as the path exists, without requiring existence."""
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(str(path)))
