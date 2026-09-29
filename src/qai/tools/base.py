"""Tool base classes, registry and the approval gate."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from qai.backend import ToolCall
from qai.workspace import Workspace


@dataclass(slots=True)
class ToolResult:
    """Outcome of a single tool invocation, rendered back to the model."""

    content: str
    ok: bool = True

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(content=message, ok=False)

    def to_message(self) -> str:
        return self.content


class Tool(Protocol):
    name: str
    description: str
    mutating: bool
    parameters: dict[str, Any]

    def run(self, args: dict[str, Any], workspace: Workspace) -> ToolResult: ...


@dataclass(slots=True)
class Request:
    """A tool call waiting for the user's decision."""

    call: ToolCall
    summary: str
    diff: str | None = None


Approver = Callable[[Request], str]
"""Returns ``"yes"``, ``"no"`` or ``"always"`` (which disables later prompts)."""


@dataclass
class ToolRegistry:
    workspace: Workspace
    read_only: bool = False
    approver: Approver | None = None
    auto_approve: bool = False
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: Sequence[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schema(self) -> list[dict[str, Any]]:
        """OpenAI-style function schemas for the chat template."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.error(
                f"unknown tool {call.name!r}; available tools: {', '.join(self.names)}"
            )

        if tool.mutating and self.read_only:
            return ToolResult.error(f"tool {tool.name!r} is disabled in read-only mode")

        if tool.mutating and not (self.auto_approve or not self.approver):
            summary, diff = self._describe(tool, call)
            request = Request(call=call, summary=summary, diff=diff)
            if self.approver is not None:
                verdict = self.approver(request)
                if verdict == "always":
                    self.auto_approve = True
                elif verdict != "yes":
                    return ToolResult.error(
                        f"the user rejected {call.name}. Do not retry it; "
                        "ask what they would prefer instead."
                    )

        try:
            return tool.run(call.arguments, self.workspace)
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            return ToolResult.error(f"{call.name} failed: {exc}")

    def _describe(self, tool: Tool, call: ToolCall) -> tuple[str, str | None]:
        describe = getattr(tool, "describe", None)
        if callable(describe):
            result = describe(call.arguments, self.workspace)
            if isinstance(result, tuple):
                return result
            return result, None
        return f"{call.name}({json.dumps(call.arguments, default=str)})", None
