"""Rich rendering, approval prompts and the interactive REPL."""

from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from qai.backend import Message, ToolCall
from qai.config import config_dir
from qai.tools.base import Request

HISTORY_LIMIT = 2000


def make_console() -> Console:
    return Console(highlight=False, soft_wrap=False)


def history_path() -> Path:
    path = config_dir() / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_session() -> PromptSession:
    return PromptSession(history=FileHistory(str(history_path())))


def render_diff(diff: str) -> Panel:
    lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            style = "bold"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        else:
            style = "dim"
        lines.append(f"[{style}]{line}[/{style}]")
    return Panel("\n".join(lines) or "(no changes)", title="diff", border_style="yellow")


def approval_prompt(console: Console, request: Request, read_only: bool) -> str:
    """Ask about a write. Returns ``"yes"``, ``"no"`` or ``"always"``."""
    if read_only:
        console.print("[red]read-only mode: tool denied[/red]")
        return "no"
    console.print(f"[bold yellow]Approve[/bold yellow] {request.summary}")
    if request.diff:
        console.print(render_diff(request.diff))
    try:
        answer = input("  [y/N/a(lways)] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "no"
    if answer in {"a", "always"}:
        return "always"
    return "yes" if answer in {"y", "yes"} else "no"


def print_response(console: Console, text: str) -> None:
    if not text:
        return
    if "```" in text:
        console.print(Markdown(text))
    else:
        console.print(Markdown(text))


def print_tool_call(console: Console, call: ToolCall) -> None:
    args = ", ".join(f"{k}={_short(v)}" for k, v in call.arguments.items())
    console.print(f"[cyan]tool[/cyan] [bold]{call.name}[/bold]({args})")


def print_tool_result(console: Console, call: ToolCall, content: str, ok: bool) -> None:
    style = "green" if ok else "red"
    label = "ok" if ok else "error"
    preview = content if len(content) <= 600 else content[:600] + "\n..."
    body = "\n".join(f"  {line}" for line in preview.splitlines())
    console.print(f"[{style}]{label}[/{style}] [dim]{call.name}[/dim]\n{body}")


def _short(value: Any, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


HELP_TEXT = """\
commands
  /help              show this help
  /new               start a fresh conversation
  /context [path]    re-send the workspace manifest, or attach paths
  /tools             list available tools
  /stats             context usage, budget and compaction count
  /model             show the active model and context settings
  /read-only on|off  toggle write access
  /exit              quit (ctrl-d also works)

anywhere in a prompt
  @src/app.py        attach a file or directory to the conversation
"""


def print_banner(console: Console, model_file: str, root: Path, read_only: bool) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("model", model_file)
    table.add_row("workspace", str(root))
    table.add_row("mode", "read-only" if read_only else "read-write")
    console.print(Panel(table, title="qai", subtitle="ctrl-d to exit", border_style="cyan"))
    console.print("[dim]/help for commands, @file to attach, ctrl-c to interrupt[/dim]\n")


def print_messages(console: Console, messages: list[Message]) -> None:
    for message in messages:
        if message.role == "user":
            console.print(f"[bold blue]>[/bold blue] {message.content}")
        elif message.content:
            console.print(Markdown(message.content))


def show_file(console: Console, path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        console.print(f"[red]cannot read {path}: {exc}[/red]")
        return
    lexer = Syntax.guess_lexer(str(path), source)
    console.print(Syntax(source[:20_000], lexer, line_numbers=True, word_wrap=False))


def diff_preview(old: str, new: str, label: str = "") -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
        )
    )


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)
