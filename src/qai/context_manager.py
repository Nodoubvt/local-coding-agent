"""Context window management: token budgeting, trimming and auto-compaction.

Three cooperating layers keep a 32k buffer usable:

1. *Budgeting* - every message is measured with a conservative char-per-token
   heuristic so the agent can reserve room for the reply and tool schemas.
2. *Trimming* - oversized tool results and context blocks are shrunk in place
   (head + tail, so errors at the end survive) before anything is dropped.
3. *Compaction* - when trimming is not enough, older turns are summarised into
   a single ``<summary>`` block, optionally via the model itself.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from qai.backend import Message

CHARS_PER_TOKEN = 3.5
"""Conservative average for code + prose with a BPE tokenizer."""

CONTEXT_RE = re.compile(r"^\s*<context>.*?</context>\s*$", re.DOTALL)
SUMMARY_OPEN, SUMMARY_CLOSE = "<summary>", "</summary>"


def estimate_tokens(text: str) -> int:
    """Rough token count. Deliberately over-estimates rather than under-estimates."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def message_tokens(message: Message, per_message_overhead: int = 8) -> int:
    total = per_message_overhead + estimate_tokens(message.content)
    for call in message.tool_calls or []:
        total += estimate_tokens(str(call))
    return total


def truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of *text*, dropping the middle."""
    if len(text) <= limit or limit <= 0:
        return text
    marker = f"\n... [{len(text) - limit} characters truncated] ...\n"
    if limit <= len(marker) + 40:
        return text[:limit]
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return text[:head] + marker + (text[-tail:] if tail else "")


@dataclass(slots=True)
class ContextPolicy:
    """Tunables for the context optimizer."""

    n_ctx: int = 32_768
    reserve_output: int = 4_096
    tool_schema_reserve: int = 1_200
    compact_threshold: float = 0.70
    max_tool_result_chars: int = 8_000
    max_context_block_chars: int = 24_000
    keep_recent_blocks: int = 6
    min_block_chars: int = 400
    per_message_overhead: int = 8

    @property
    def budget(self) -> int:
        return max(1_000, self.n_ctx - self.reserve_output - self.tool_schema_reserve)

    @property
    def soft_budget(self) -> int:
        return int(self.budget * self.compact_threshold)


@dataclass
class Block:
    """An indivisible group of messages that must be kept or dropped together.

    Chat templates require every assistant message that carries ``tool_calls`` to
    be followed by the matching ``tool`` messages, so those are grouped into one
    block and can never be split or orphaned.
    """

    messages: list[Message]
    kind: str = "turn"
    pinned: bool = False

    @property
    def tokens(self) -> int:
        return sum(message_tokens(m, 8) for m in self.messages)


@dataclass
class CompactReport:
    """What a compaction pass actually did, for logging and tests."""

    dropped_blocks: int = 0
    dropped_tokens: int = 0
    summary: str = ""
    used_fallback: bool = False

    @property
    def did_anything(self) -> bool:
        return bool(self.summary) or self.dropped_blocks > 0


def build_blocks(messages: list[Message], policy: ContextPolicy) -> list[Block]:
    """Group *messages* into atomic blocks, pinning the system prompt."""
    blocks: list[Block] = []
    index = 0

    while index < len(messages):
        message = messages[index]

        if message.role == "system":
            blocks.append(Block([_copy(message)], kind="system", pinned=True))
            index += 1
            continue

        if message.role == "assistant" and message.tool_calls:
            group = [_copy(message)]
            index += 1
            while index < len(messages) and messages[index].role == "tool":
                group.append(_copy(messages[index]))
                index += 1
            blocks.append(Block(group, kind="tools"))
            continue

        if message.role == "user" and CONTEXT_RE.match(message.content or ""):
            blocks.append(Block([_copy(message)], kind="context"))
            index += 1
            continue

        blocks.append(Block([_copy(message)], kind=message.role))
        index += 1

    return blocks


def _copy(message: Message) -> Message:
    return Message(
        role=message.role,
        content=message.content,
        tool_calls=[dict(call) for call in (message.tool_calls or [])] or None,
        name=message.name,
        tool_call_id=message.tool_call_id,
    )


def usage(messages: list[Message], policy: ContextPolicy) -> int:
    return sum(message_tokens(m, policy.per_message_overhead) for m in messages)


def optimize(
    messages: list[Message],
    policy: ContextPolicy,
    *,
    keep_last: int = 1,
) -> list[Message]:
    """Fit *messages* into the budget without losing tool-call pairing.

    Shrinks oversized content first, then drops the oldest droppable blocks, then
    hard-truncates whatever is left. The final *keep_last* blocks are never
    dropped, because they hold the turn currently in flight.
    """
    blocks = build_blocks(messages, policy)
    if not blocks:
        return []

    _shrink(blocks, policy)
    total = sum(block.tokens for block in blocks)

    if total <= policy.budget:
        return [m for block in blocks for m in block.messages]

    # Drop oldest droppable blocks, never touching the in-flight turn.
    protected_from = max(0, len(blocks) - keep_last)
    index = 0
    while total > policy.budget and index < protected_from:
        block = blocks[index]
        if block.pinned:
            index += 1
            continue
        total -= block.tokens
        blocks[index] = Block([], kind="dropped")
        index += 1

    if total <= policy.budget:
        return [m for block in blocks if block.messages for m in block.messages]

    # Still too large: truncate the biggest remaining non-system blocks.
    survivors = [b for b in blocks if b.messages]
    while survivors and total > policy.budget:
        victim = max(
            (b for b in survivors if not b.pinned),
            key=lambda b: b.tokens,
            default=None,
        )
        if victim is None:
            break
        overflow = total - policy.budget
        saved = _truncate_block(victim, overflow, policy)
        if saved <= 0:
            break
        total -= saved

    return [m for block in blocks if block.messages for m in block.messages]


def _shrink(blocks: list[Block], policy: ContextPolicy) -> None:
    for block in blocks:
        for message in block.messages:
            if message.role == "tool":
                message.content = truncate_middle(message.content, policy.max_tool_result_chars)
            elif message.role == "user" and CONTEXT_RE.match(message.content or ""):
                message.content = truncate_middle(
                    message.content, policy.max_context_block_chars
                )


def _truncate_block(block: Block, overflow: int, policy: ContextPolicy) -> int:
    """Cut roughly *overflow* tokens out of *block*. Returns tokens saved."""
    target = max(policy.min_block_chars, int(overflow * CHARS_PER_TOKEN))
    before = block.tokens

    for message in reversed(block.messages):
        if not message.content:
            continue
        message.content = truncate_middle(message.content, max(policy.min_block_chars, 200))
        if before - block.tokens >= overflow:
            break

    for message in reversed(block.messages):
        if not message.content:
            continue
        current = len(message.content)
        allowance = max(0, target - (before - block.tokens - estimate_tokens(message.content)))
        if current > allowance:
            message.content = truncate_middle(message.content, allowance)
            if before - block.tokens >= overflow:
                break

    return max(0, before - block.tokens)


def _render(messages: list[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.role
        if role == "tool":
            role = f"tool:{message.name or 'unknown'}"
        body = (message.content or "").strip()
        if not body:
            continue
        lines.append(f"[{role}] {body}")
    return "\n\n".join(lines)


def extractive_summary(messages: list[Message], per_message: int = 200) -> str:
    """Deterministic fallback summary used when no summarizer is available."""
    parts: list[str] = []
    for message in messages:
        body = " ".join((message.content or "").split())
        if not body:
            continue
        label = "user" if message.role == "user" else message.role
        parts.append(f"- {label}: {body[:per_message]}")
    if not parts:
        return ""
    joined = "\n".join(parts)
    return truncate_middle(joined, 2_000)


Summarizer = Callable[[str], str]


def compact(
    messages: list[Message],
    policy: ContextPolicy,
    *,
    summarizer: Summarizer | None = None,
) -> tuple[list[Message], CompactReport]:
    """Fold older turns into one summary block, newest turns kept verbatim."""
    blocks = build_blocks(messages, policy)
    if len(blocks) <= policy.keep_recent_blocks:
        return messages, CompactReport()

    tail = blocks[-policy.keep_recent_blocks :]
    head = blocks[: -policy.keep_recent_blocks]
    system = [b for b in head if b.pinned]
    older = [b for b in head if not b.pinned]

    if not older:
        return messages, CompactReport()

    transcript = _render([m for b in older for m in b.messages])
    report = CompactReport(dropped_blocks=len(older), dropped_tokens=sum(b.tokens for b in older))
    summary = ""

    if summarizer and transcript:
        try:
            summary = (summarizer(transcript) or "").strip()
        except Exception:  # noqa: BLE001 - a failed summary must not kill the session
            summary = ""
        if summary:
            summary = f"{SUMMARY_OPEN}\n{summary}\n{SUMMARY_CLOSE}"
        else:
            report.used_fallback = True
    else:
        report.used_fallback = True

    if not summary:
        summary_body = extractive_summary([m for b in older for m in b.messages])
        if not summary_body:
            return messages, CompactReport()
        summary = f"{SUMMARY_OPEN}\nextractive\n{summary_body}\n{SUMMARY_CLOSE}"

    report.summary = summary

    rebuilt: list[Message] = [m for b in system for m in b.messages]
    rebuilt.append(Message(role="user", content=summary))
    rebuilt.extend(m for b in tail for m in b.messages)
    return rebuilt, report


class ContextManager:
    """Stateful facade the agent talks to between turns."""

    def __init__(self, policy: ContextPolicy, auto_compact: bool = True) -> None:
        self.policy = policy
        self.auto_compact = auto_compact
        self.compaction_count = 0
        self.last_report = CompactReport()

    def ratio(self, messages: list[Message]) -> float:
        return usage(messages, self.policy) / self.policy.budget

    def should_compact(self, messages: list[Message]) -> bool:
        return self.auto_compact and self.ratio(messages) > self.policy.compact_threshold

    def fit(self, messages: list[Message], keep_last: int = 1) -> list[Message]:
        return optimize(messages, self.policy, keep_last=keep_last)

    def maybe_compact(
        self,
        messages: list[Message],
        summarizer: Summarizer | None = None,
    ) -> list[Message]:
        if not self.should_compact(messages):
            return messages
        compacted, report = compact(messages, self.policy, summarizer=summarizer)
        if report.did_anything:
            self.compaction_count += 1
            self.last_report = report
        return compacted

    def stats(self, messages: list[Message]) -> dict[str, Any]:
        return {
            "n_ctx": self.policy.n_ctx,
            "budget": self.policy.budget,
            "used_tokens": usage(messages, self.policy),
            "ratio": round(self.ratio(messages), 3),
            "messages": len(messages),
            "compactions": self.compaction_count,
        }
