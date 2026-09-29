from __future__ import annotations

from qai.backend import Message
from qai.context_manager import (
    ContextManager,
    ContextPolicy,
    build_blocks,
    compact,
    estimate_tokens,
    extractive_summary,
    optimize,
    truncate_middle,
    usage,
)


def policy(**kwargs) -> ContextPolicy:
    base = dict(
        n_ctx=8_000,
        reserve_output=1_000,
        tool_schema_reserve=200,
        compact_threshold=0.7,
        max_tool_result_chars=2_000,
        max_context_block_chars=4_000,
        keep_recent_blocks=3,
    )
    base.update(kwargs)
    return ContextPolicy(**base)


def tool_turn(name: str, result: str) -> list[Message]:
    return [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": name, "arguments": {}},
                }
            ],
        ),
        Message(role="tool", name=name, tool_call_id="c1", content=result),
    ]


def test_estimate_tokens_is_conservative():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 350) >= 100


def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD" + "x" * 5_000 + "TAIL"
    out = truncate_middle(text, 400)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "truncated" in out


def test_truncate_middle_noop_when_small():
    assert truncate_middle("short", 100) == "short"


def test_budget_reserves_output_and_schema():
    p = policy()
    assert p.budget == 8_000 - 1_000 - 200
    assert p.soft_budget < p.budget


def test_blocks_keep_tool_calls_paired():
    messages = [
        Message(role="system", content="sys"),
        *tool_turn("read_file", "content"),
        Message(role="assistant", content="done"),
    ]
    blocks = build_blocks(messages, policy())
    kinds = [b.kind for b in blocks]
    assert kinds == ["system", "tools", "assistant"]
    tool_block = blocks[1]
    assert len(tool_block.messages) == 2
    assert tool_block.messages[0].role == "assistant"
    assert tool_block.messages[1].role == "tool"


def test_optimize_trims_tool_results():
    p = policy(max_tool_result_chars=500)
    messages = [Message(role="system", content="sys"), *tool_turn("read_file", "y" * 20_000)]
    out = optimize(messages, p)
    tool_message = next(m for m in out if m.role == "tool")
    assert len(tool_message.content) <= 600
    assert "truncated" in tool_message.content


def test_optimize_drops_old_turns_but_keeps_system():
    p = policy(n_ctx=2_000, reserve_output=200, tool_schema_reserve=0)
    messages = [Message(role="system", content="sys")]
    for i in range(40):
        messages.append(Message(role="user", content=f"question {i} " + "z" * 400))
        messages.append(Message(role="assistant", content=f"answer {i} " + "z" * 400))
    messages.append(Message(role="user", content="the current question"))

    out = optimize(messages, p, keep_last=1)
    assert out[0].role == "system"
    assert out[-1].content == "the current question"
    assert usage(out, p) <= p.budget


def test_optimize_never_orphans_tool_messages():
    p = policy(n_ctx=2_500, reserve_output=200, tool_schema_reserve=0)
    messages = [Message(role="system", content="sys")]
    for i in range(20):
        messages.extend(tool_turn("read_file", f"result {i} " + "q" * 500))
    out = optimize(messages, p, keep_last=1)

    pending = False
    for message in out:
        if message.role == "assistant" and message.tool_calls:
            pending = True
        elif message.role == "tool":
            assert pending, "tool message without a preceding assistant tool_call"
            pending = False
        elif pending:
            raise AssertionError("assistant tool_calls not followed by a tool result")


def test_optimize_keeps_recent_tool_turn_even_when_oversized():
    p = policy(n_ctx=1_500, reserve_output=200, tool_schema_reserve=0, max_tool_result_chars=1_000)
    messages = [Message(role="system", content="sys"), *tool_turn("read_file", "w" * 40_000)]
    out = optimize(messages, p, keep_last=1)
    assert any(m.role == "tool" for m in out)
    assert out[-1].role == "tool"


def test_optimize_preserves_context_block_shape():
    p = policy()
    block = "<context>\n" + "f" * 5_000 + "\n</context>"
    out = optimize([Message(role="system", content="s"), Message(role="user", content=block)], p)
    user = next(m for m in out if m.role == "user")
    assert user.content.startswith("<context>")


def test_compact_is_noop_when_short():
    messages = [Message(role="system", content="s"), Message(role="user", content="hi")]
    out, report = compact(messages, policy())
    assert out == messages
    assert not report.did_anything


def test_compact_summarizes_old_turns():
    p = policy(keep_recent_blocks=2)
    messages = [Message(role="system", content="sys")]
    for i in range(10):
        messages.append(Message(role="user", content=f"u{i} " + "a" * 300))
        messages.append(Message(role="assistant", content=f"a{i} " + "b" * 300))
    messages.append(Message(role="user", content="latest"))
    messages.append(Message(role="assistant", content="latest answer"))

    out, report = compact(messages, p, summarizer=lambda t: "earlier work: refactor")

    assert report.dropped_blocks > 0
    assert report.summary.startswith("<summary>")
    assert "refactor" in report.summary
    assert out[-1].content == "latest answer"
    assert usage(out, p) < usage(messages, p)


def test_compact_falls_back_to_extractive():
    p = policy(keep_recent_blocks=1)
    messages = [Message(role="system", content="sys")]
    for i in range(8):
        messages.append(Message(role="user", content=f"u{i} " + "c" * 300))
        messages.append(Message(role="assistant", content=f"a{i} " + "d" * 300))
    messages.append(Message(role="user", content="latest"))

    out, report = compact(messages, p, summarizer=lambda t: "")
    assert report.used_fallback
    assert any(m.content.startswith("<summary>") for m in out)


def test_compact_survives_summarizer_exception():
    p = policy(keep_recent_blocks=1)
    messages = [Message(role="system", content="sys")]
    for i in range(8):
        messages.append(Message(role="user", content=f"u{i} " + "e" * 300))
        messages.append(Message(role="assistant", content=f"a{i} " + "f" * 300))
    messages.append(Message(role="user", content="latest"))

    def boom(_text):
        raise RuntimeError("model exploded")

    out, report = compact(messages, p, summarizer=boom)
    assert report.used_fallback
    assert out[-1].content == "latest"


def test_compact_keeps_system_prompt():
    p = policy(keep_recent_blocks=1)
    messages = [Message(role="system", content="sys prompt")]
    for i in range(8):
        messages.append(Message(role="user", content=f"u{i}"))
        messages.append(Message(role="assistant", content=f"a{i}"))
    messages.append(Message(role="user", content="latest"))
    out, _ = compact(messages, p)
    assert out[0].role == "system"
    assert out[0].content == "sys prompt"


def test_extractive_summary_caps_length():
    messages = [Message(role="user", content="x" * 10_000) for _ in range(10)]
    summary = extractive_summary(messages)
    assert len(summary) <= 2_100


def test_context_manager_threshold_and_count():
    p = policy(n_ctx=4_000, reserve_output=200, tool_schema_reserve=0, compact_threshold=0.5)
    manager = ContextManager(p, auto_compact=True)

    small = [Message(role="system", content="sys"), Message(role="user", content="hi")]
    assert not manager.should_compact(small)
    assert manager.maybe_compact(small) is small
    assert manager.compaction_count == 0

    big = [Message(role="system", content="sys")]
    for i in range(30):
        big.append(Message(role="user", content=f"u{i} " + "g" * 300))
        big.append(Message(role="assistant", content=f"a{i} " + "h" * 300))
    assert manager.should_compact(big)
    after = manager.maybe_compact(big, summarizer=lambda t: "summary text")
    assert manager.compaction_count == 1
    assert manager.ratio(after) < manager.ratio(big)


def test_context_manager_auto_compact_disabled():
    manager = ContextManager(policy(), auto_compact=False)
    big = [Message(role="system", content="sys")]
    for i in range(30):
        big.append(Message(role="user", content=f"u{i} " + "g" * 300))
    assert not manager.should_compact(big)
    assert manager.maybe_compact(big) is big


def test_context_manager_fit_always_bounds():
    manager = ContextManager(policy(n_ctx=2_000, reserve_output=200, tool_schema_reserve=0))
    messages = [Message(role="system", content="s")]
    for i in range(50):
        messages.append(Message(role="user", content=f"u{i} " + "k" * 500))
    fitted = manager.fit(messages, keep_last=1)
    assert usage(fitted, manager.policy) <= manager.policy.budget


def test_stats_reports_usage():
    manager = ContextManager(policy())
    stats = manager.stats([Message(role="user", content="hello world")])
    assert stats["n_ctx"] == 8_000
    assert stats["used_tokens"] > 0
    assert 0 < stats["ratio"] < 1


def test_thirty_two_k_default_budget():
    p = ContextPolicy(n_ctx=32_768, reserve_output=4_096, tool_schema_reserve=1_200)
    assert p.budget == 32_768 - 4_096 - 1_200
    assert p.budget == 27_472
