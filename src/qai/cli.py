"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qai import __version__
from qai.agent import Agent, policy_from_settings
from qai.backend import LlamaBackend, Message, ModelUnavailableError
from qai.config import Settings, load_settings, model_cache_dir
from qai.context import collect_context, extract_mentions
from qai.context_manager import ContextManager
from qai.tools import ToolRegistry, default_tools
from qai.ui import (
    HELP_TEXT,
    approval_prompt,
    make_console,
    make_session,
    print_banner,
    print_response,
    print_tool_call,
    print_tool_result,
)
from qai.workspace import Workspace

app = typer.Typer(
    add_completion=False,
    help="A tiny local coding agent powered by Qwen2.5-Coder 1.5B.",
    no_args_is_help=False,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"qai {__version__}")
        raise typer.Exit()


@app.command()
def main(
    ctx: typer.Context,
    prompt: Annotated[
        str | None, typer.Argument(help="Prompt. Omit for an interactive session.")
    ] = None,
    attach: Annotated[
        list[str] | None,
        typer.Option("--attach", "-a", help="Attach a file, directory or glob to the prompt."),
    ] = None,
    cwd: Annotated[
        Path, typer.Option("--cwd", "-C", help="Workspace root (default: .)")
    ] = Path("."),
    ignore: Annotated[
        list[str] | None,
        typer.Option("--ignore", "-i", help="Extra ignore pattern to hide, repeatable."),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="GGUF file name or path.")
    ] = None,
    repo: Annotated[
        str | None, typer.Option("--repo", help="Hugging Face GGUF repo id.")
    ] = None,
    n_ctx: Annotated[
        int | None, typer.Option("--ctx-size", help="Context window (default 32768).")
    ] = None,
    gpu_layers: Annotated[
        int | None,
        typer.Option("--gpu-layers", help="-1 offloads everything, 0 is CPU only."),
    ] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", "-t")
    ] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    max_iterations: Annotated[
        int | None, typer.Option("--max-iterations")
    ] = None,
    system: Annotated[
        str | None, typer.Option("--system", "-s", help="Override the system prompt.")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Auto-approve all writes.")
    ] = False,
    read_only: Annotated[
        bool, typer.Option("--read-only", help="Refuse every write tool.")
    ] = False,
    no_stream: Annotated[
        bool, typer.Option("--no-stream", help="Do not stream output.")
    ] = False,
    no_tools: Annotated[
        bool, typer.Option("--no-tools", help="Chat only; never touch the filesystem.")
    ] = False,
    no_compact: Annotated[
        bool, typer.Option("--no-compact", help="Disable automatic context compaction.")
    ] = False,
    compact_threshold: Annotated[
        float | None,
        typer.Option("--compact-at", help="Auto-compact above this context ratio (0-1)."),
    ] = None,
    max_tool_chars: Annotated[
        int | None,
        typer.Option("--max-tool-chars", help="Trim tool results to this many characters."),
    ] = None,
    print_config: Annotated[
        bool, typer.Option("--print-config", help="Show resolved settings and exit.")
    ] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=version_callback, is_eager=True)
    ] = False,
) -> None:
    settings = load_settings(
        {
            "model_file": _resolve_model_arg(model),
            "model_repo": repo,
            "n_ctx": n_ctx,
            "n_gpu_layers": gpu_layers,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_iterations": max_iterations,
            "system_prompt": system,
            "auto_approve": True if yes else None,
            "stream": False if no_stream else None,
            "read_only": True if read_only else None,
            "auto_compact": False if no_compact else None,
            "compact_threshold": compact_threshold,
            "max_tool_result_chars": max_tool_chars,
        }
    )

    console = make_console()
    if print_config:
        console.print(Markdown("```json\n" + _pretty(settings.to_dict()) + "\n```"))
        raise typer.Exit

    try:
        workspace = Workspace(cwd)
    except OSError as exc:
        console.print(f"[red]cannot use workspace {cwd}: {exc}[/red]")
        raise typer.Exit(2) from exc

    if ignore:
        workspace.add_ignores(ignore)

    registry = ToolRegistry(
        workspace=workspace,
        read_only=settings.read_only,
        auto_approve=settings.auto_approve,
        approver=lambda request: approval_prompt(console, request, settings.read_only),
    )
    if not no_tools:
        registry.register_all(default_tools())

    agent_kwargs: dict[str, Any] = {}
    if not no_tools:
        agent_kwargs = {
            "on_tool_start": lambda call: print_tool_call(console, call),
            "on_tool_result": lambda call, body, ok: print_tool_result(
                console, call, body, ok
            ),
            "on_compact": lambda summary: console.print(
                f"[dim]context compacted, freed {len(summary)} chars of older turns[/dim]"
            ),
        }

    try:
        backend = LlamaBackend(settings)
    except ModelUnavailableError as exc:
        console.print(Panel(Text(str(exc)), title="model unavailable", border_style="red"))
        raise typer.Exit(1) from exc

    agent = Agent(
        backend=backend,
        registry=registry,
        max_iterations=settings.max_iterations,
        system_prompt=settings.system_prompt,
        context=ContextManager(
            policy_from_settings(settings), auto_compact=settings.auto_compact
        ),
        **agent_kwargs,
    )

    if prompt is not None:
        _run_once(console, agent, settings, workspace, prompt, list(attach or []), no_tools)
        return

    _repl(console, agent, settings, workspace, registry, no_tools)


def _run_once(
    console,
    agent: Agent,
    settings: Settings,
    workspace: Workspace,
    prompt: str,
    attach: list[str],
    no_tools: bool,
) -> None:
    cleaned, mentions = extract_mentions(prompt)
    files = [*mentions, *attach]
    context = collect_context(workspace, files, include_manifest=not no_tools)

    messages: list[Message] = []
    if context:
        messages.append(Message(role="user", content=f"<context>\n{context}\n</context>"))
    messages.append(Message(role="user", content=cleaned.strip() or prompt))

    if settings.stream and not console.is_terminal:
        response = agent.run(messages, stream=None, tools_enabled=not no_tools)
        print_response(console, response.text)
    elif settings.stream:
        printed = {"any": False}

        def _emit(piece: str) -> None:
            if not printed["any"]:
                console.file.write(piece)
                printed["any"] = True
            else:
                console.file.write(piece)
            console.file.flush()

        try:
            response = agent.run(messages, stream=_emit, tools_enabled=not no_tools)
        finally:
            if printed["any"]:
                console.file.write("\n")
                console.file.flush()
        if not printed["any"]:
            print_response(console, response.text)
    else:
        response = agent.run(messages, stream=None, tools_enabled=not no_tools)
        print_response(console, response.text)


def _repl(
    console,
    agent: Agent,
    settings: Settings,
    workspace: Workspace,
    registry: ToolRegistry,
    no_tools: bool,
) -> None:
    print_banner(console, settings.model_file, workspace.root, settings.read_only)
    session = make_session()
    history: list[Message] = []

    while True:
        try:
            raw = session.prompt("qai> ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return

        raw = raw.strip()
        if not raw:
            continue
        if raw in {"/exit", "/quit"}:
            return
        if raw == "/help":
            console.print(Markdown(HELP_TEXT))
            continue
        if raw == "/new":
            history.clear()
            console.print("[dim]conversation cleared[/dim]")
            continue
        if raw == "/tools":
            names = registry.names or ["(none)"]
            console.print(f"[bold]tools:[/bold] {', '.join(names)}")
            continue
        if raw in {"/stats", "/context-stats"}:
            stats = agent.context.stats([Message(role="user", content=x) for x in _flat(history)])
            table = Table.grid(padding=(0, 2))
            for key, value in stats.items():
                table.add_row(f"[bold]{key}[/bold]", str(value))
            console.print(Panel(table, title="context", border_style="cyan"))
            continue
        if raw == "/model":
            console.print(
                f"[bold]model[/bold] {settings.model_repo}/{settings.model_file}\n"
                f"[bold]ctx[/bold] {settings.n_ctx}  [bold]temp[/bold] {settings.temperature}"
            )
            continue
        if raw.startswith("/context"):
            parts = raw.split(maxsplit=1)
            targets = [parts[1]] if len(parts) > 1 else []
            block = collect_context(workspace, targets)
            history.append(Message(role="user", content=f"<context>\n{block}\n</context>"))
            console.print(f"[dim]context updated ({len(targets) or 'manifest'})[/dim]")
            continue
        if raw.startswith("/read-only"):
            parts = raw.split(maxsplit=1)
            value = parts[1].strip().lower() if len(parts) > 1 else "on"
            settings.read_only = value in {"on", "true", "yes", "y", "1"}
            registry.read_only = settings.read_only
            console.print(f"[dim]read-only = {settings.read_only}[/dim]")
            continue

        cleaned, mentions = extract_mentions(raw)
        if not no_tools:
            block = collect_context(workspace, mentions)
            if block:
                history.append(Message(role="user", content=f"<context>\n{block}\n</context>"))

        try:
            response = agent.run(
                [*history, Message(role="user", content=cleaned.strip() or raw)],
                tools_enabled=not no_tools,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/yellow]")
            continue

        print_response(console, response.text)
        history = adopt_history(agent, history)
        history.append(Message(role="user", content=cleaned.strip() or raw))
        history.append(Message(role="assistant", content=response.text))


def adopt_history(agent: Agent, previous: list[Message]) -> list[Message]:
    """Carry the agent's trimmed/compacted transcript into the next REPL turn.

    Trailing assistant/tool turns belong to the turn that just finished, and the
    caller re-appends the (user, assistant) pair, so both are dropped here.
    """
    if not agent.last_history:
        return previous
    adopted = [m for m in agent.last_history if m.role != "system"]
    while adopted and adopted[-1].role in {"assistant", "tool", "user"}:
        adopted.pop()
    return adopted or previous


def _flat(history: list[Message]) -> list[str]:
    return [m.content for m in history if m.content]


def _resolve_model_arg(model: str | None) -> str | None:
    if not model:
        return None
    candidate = Path(model).expanduser()
    if candidate.suffix == ".gguf" and candidate.is_file():
        target = model_cache_dir() / candidate.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(target)
        return target.name
    return model


def _pretty(data: dict) -> str:
    return json.dumps(data, indent=2)


def run() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    run()
