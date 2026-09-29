"""Agent loop: prompt -> tool calls -> results -> final answer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qai.backend import LlamaBackend, Message, Response, ToolCall
from qai.context_manager import ContextManager, ContextPolicy
from qai.tools.base import ToolRegistry

StreamHandler = Callable[[str], None]

GUIDANCE = (
    "To use a tool, reply with a single fenced block and nothing else:\n"
    '```json\n{"tool": "read_file", "arguments": {"path": "src/app.py"}}\n```\n'
    "Call one tool at a time and wait for its result. When the task is done, "
    "reply with plain text and no code fence."
)

SUMMARIZE_PROMPT = (
    "Summarize the following transcript of a coding session for your own future "
    "reference. Keep file paths, function names, decisions, edits made, and open "
    "questions. Drop file dumps and tool noise. Answer with under 300 words of "
    "plain prose or bullets, no preamble.\n\n"
)


@dataclass
class Agent:
    """Drives the model/tool loop until the model stops calling tools."""

    backend: LlamaBackend
    registry: ToolRegistry
    max_iterations: int = 12
    system_prompt: str = ""
    on_tool_start: Callable[[ToolCall], None] | None = None
    on_tool_result: Callable[[ToolCall, str, bool], None] | None = None
    on_compact: Callable[[str], None] | None = None
    context: ContextManager = field(
        default_factory=lambda: ContextManager(ContextPolicy())
    )
    last_history: list[Message] = field(default_factory=list)
    """The post-compaction transcript from the most recent :meth:`run`.

    The REPL adopts this so that trimming and summarization survive between
    turns instead of being thrown away at the end of every one.
    """

    def run(
        self,
        messages: list[Message],
        stream: StreamHandler | None = None,
        tools_enabled: bool = True,
    ) -> Response:
        system = self.system_prompt
        if tools_enabled:
            system = f"{system}\n\n{GUIDANCE}"
        history: list[Message] = [Message(role="system", content=system), *messages]
        schema = self.registry.schema() if tools_enabled else None

        try:
            for _ in range(self.max_iterations):
                history = self._budget(history, keep_last=1)
                response = self._complete(history, schema, stream)
                if not response.tool_calls:
                    return response

                history.append(
                    Message(
                        role="assistant",
                        content=response.text,
                        tool_calls=[
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }
                            for call in response.tool_calls
                        ],
                    )
                )

                for call in response.tool_calls:
                    if self.on_tool_start:
                        self.on_tool_start(call)
                    result = self.registry.dispatch(call)
                    if self.on_tool_result:
                        self.on_tool_result(call, result.content, result.ok)
                    history.append(
                        Message(
                            role="tool",
                            name=call.name,
                            tool_call_id=call.id,
                            content=result.to_message(),
                        )
                    )

            return Response(
                text=(
                    "Stopped after reaching the step limit "
                    f"({self.max_iterations}). Re-run with --max-iterations to continue."
                ),
                finish_reason="length",
            )
        finally:
            self.last_history = history

    def _budget(self, history: list[Message], keep_last: int = 1) -> list[Message]:
        """Auto-compact, then hard-fit whatever is still over budget."""
        before = self.context.compaction_count
        history = self.context.maybe_compact(history, summarizer=self._summarize)
        if self.on_compact and self.context.compaction_count > before:
            self.on_compact(self.context.last_report.summary)
        return self.context.fit(history, keep_last=keep_last)

    def _summarize(self, transcript: str) -> str:
        """One-shot, low-temperature summary request used by auto-compact."""
        request = [
            Message(role="system", content=SUMMARIZE_PROMPT),
            Message(role="user", content=transcript),
        ]
        try:
            return self.backend.complete(
                request,
                None,
                max_tokens_override=512,
            ).text
        except Exception:  # noqa: BLE001 - fall back to extractive summary
            return ""

    def _complete(
        self,
        history: list[Message],
        schema: list[dict[str, Any]] | None,
        stream: StreamHandler | None,
    ) -> Response:
        # Tool calls must be read from a non-streamed completion: llama.cpp
        # delivers them in the final chunk, and a 1.5B model often emits a
        # fenced JSON block mid-stream that only normalises correctly once the
        # whole turn is available. Text is still emitted incrementally below.
        response = self.backend.complete(history, schema)
        if stream is not None and response.text:
            for piece in _chunks(response.text):
                stream(piece)
        return response


def _chunks(text: str, size: int = 48) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def policy_from_settings(settings: Any) -> ContextPolicy:
    """Build a :class:`ContextPolicy` from CLI/user settings."""
    return ContextPolicy(
        n_ctx=settings.n_ctx,
        reserve_output=max(1_024, settings.max_tokens),
        compact_threshold=settings.compact_threshold,
        max_tool_result_chars=settings.max_tool_result_chars,
        keep_recent_blocks=settings.keep_recent_blocks,
    )
