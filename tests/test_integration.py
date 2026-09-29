from __future__ import annotations

import io

from rich.console import Console

from qai.agent import Agent, policy_from_settings
from qai.backend import Message, Response, ToolCall
from qai.config import Settings
from qai.context_manager import ContextManager, usage
from qai.tools import ToolRegistry, default_tools
from qai.workspace import Workspace


class ScriptedBackend:
    def __init__(self, script, summary="earlier turns summarized"):
        self.script = list(script)
        self.summary = summary
        self.prompts = []
        self.summary_calls = 0
        self.settings = Settings(n_ctx=32_768)

    def complete(self, messages, tools=None, max_tokens_override=None):
        # Summarization is a side-channel request (max_tokens_override set);
        # answer it without consuming the scripted conversation.
        if max_tokens_override is not None:
            self.summary_calls += 1
            return Response(text=self.summary)
        self.prompts.append([m.to_dict() for m in messages])
        return self.script.pop(0) if self.script else Response(text="done")


def test_realistic_session_stays_under_32k(tmp_path):
    """A long read/edit/grep session must never overflow the buffer."""
    big = tmp_path / "big.py"
    big.write_text("\n".join(f"line_{i} = {i}" for i in range(20_000)), encoding="utf-8")
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")

    script = []
    for _ in range(14):
        script.append(
            Response(
                tool_calls=[
                    ToolCall(name="read_file", arguments={"path": "big.py"})
                ]
            )
        )
    script.append(Response(text="All done."))

    backend = ScriptedBackend(script)
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    settings = Settings()
    policy = policy_from_settings(settings)
    agent = Agent(
        backend=backend,
        registry=registry,
        max_iterations=15,
        system_prompt="sys",
        context=ContextManager(policy),
    )

    result = agent.run([Message(role="user", content="read everything")])

    assert result.text == "All done."
    policy = policy_from_settings(Settings())
    for prompt in backend.prompts:
        replay = [Message(role=m["role"], content=m.get("content") or "") for m in prompt]
        assert usage(replay, policy) <= policy.budget

    # tool-call pairing survived the trimming
    for prompt in backend.prompts:
        pending = False
        for entry in prompt:
            if entry["role"] == "assistant" and entry.get("tool_calls"):
                pending = True
            elif entry["role"] == "tool":
                assert pending
                pending = False


def test_compaction_kicks_in_during_long_session(tmp_path):
    # Each read is trimmed to max_tool_result_chars, so 25 rounds of an 8k
    # result must exceed the 27k budget and force at least one compaction.
    (tmp_path / "f.py").write_text("y = 2\n" * 5_000, encoding="utf-8")
    script = [
        Response(tool_calls=[ToolCall(name="read_file", arguments={"path": "f.py"})])
        for _ in range(25)
    ] + [Response(text="finished")]

    backend = ScriptedBackend(script)
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    settings = Settings()
    agent = Agent(
        backend=backend,
        registry=registry,
        max_iterations=30,
        system_prompt="sys",
        context=ContextManager(policy_from_settings(settings)),
    )
    agent.run([Message(role="user", content="loop")])

    assert agent.context.compaction_count >= 1


def test_tool_result_is_trimmed_before_model_sees_it(tmp_path):
    huge = tmp_path / "huge.py"
    huge.write_text("z" * 500_000, encoding="utf-8")

    backend = ScriptedBackend(
        [
            Response(tool_calls=[ToolCall(name="read_file", arguments={"path": "huge.py"})]),
            Response(text="done"),
        ]
    )
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    settings = Settings()
    agent = Agent(
        backend=backend,
        registry=registry,
        system_prompt="sys",
        context=ContextManager(policy_from_settings(settings)),
    )
    agent.run([Message(role="user", content="read it")])

    second_prompt = backend.prompts[1]
    tool_message = next(m for m in second_prompt if m["role"] == "tool")
    assert len(tool_message["content"]) <= settings.max_tool_result_chars + 100
    assert "truncated" in tool_message["content"]


def test_compaction_notice_fires_once_per_compaction(tmp_path):
    (tmp_path / "f.py").write_text("y = 2\n" * 5_000, encoding="utf-8")
    script = [
        Response(tool_calls=[ToolCall(name="read_file", arguments={"path": "f.py"})])
        for _ in range(25)
    ] + [Response(text="finished")]

    notices: list[str] = []
    backend = ScriptedBackend(script)
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    agent = Agent(
        backend=backend,
        registry=registry,
        max_iterations=30,
        system_prompt="sys",
        on_compact=notices.append,
        context=ContextManager(policy_from_settings(Settings())),
    )
    agent.run([Message(role="user", content="loop")])

    assert agent.context.compaction_count == len(notices)


def test_agent_exposes_compacted_history_for_reuse(tmp_path):
    (tmp_path / "f.py").write_text("y = 2\n" * 5_000, encoding="utf-8")
    script = [
        Response(tool_calls=[ToolCall(name="read_file", arguments={"path": "f.py"})])
        for _ in range(25)
    ] + [Response(text="finished")]

    backend = ScriptedBackend(script)
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    agent = Agent(
        backend=backend,
        registry=registry,
        max_iterations=30,
        system_prompt="sys",
        context=ContextManager(policy_from_settings(Settings())),
    )
    agent.run([Message(role="user", content="loop")])

    assert agent.last_history
    assert agent.last_history[0].role == "system"
    assert any(m.content.startswith("<summary>") for m in agent.last_history)


def test_repl_history_adoption_does_not_duplicate_turns(tmp_path):
    from qai.cli import adopt_history

    backend = ScriptedBackend([Response(text="answer one")])
    registry = ToolRegistry(workspace=Workspace(tmp_path), auto_approve=True)
    agent = Agent(backend=backend, registry=registry, system_prompt="sys")

    previous = [
        Message(role="user", content="earlier"),
        Message(role="assistant", content="earlier reply"),
    ]
    agent.run([*previous, Message(role="user", content="the question")])

    history = adopt_history(agent, previous)
    history.append(Message(role="user", content="the question"))
    history.append(Message(role="assistant", content="answer one"))

    roles = [m.role for m in history]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert [m.content for m in history].count("the question") == 1


def test_read_only_blocks_write_end_to_end(tmp_path):
    backend = ScriptedBackend(
        [
            Response(
                tool_calls=[
                    ToolCall(name="write_file", arguments={"path": "new.txt", "content": "x"})
                ]
            ),
            Response(text="cannot"),
        ]
    )
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(workspace=workspace, read_only=True, auto_approve=True)
    registry.register_all(default_tools())

    agent = Agent(backend=backend, registry=registry, system_prompt="sys")
    agent.run([Message(role="user", content="write new.txt")])

    assert not (tmp_path / "new.txt").exists()
    tool_message = next(m for m in backend.prompts[1] if m["role"] == "tool")
    assert "read-only" in tool_message["content"]


def test_diff_renders_with_color(tmp_path):
    from qai.tools.base import Request
    from qai.ui import render_diff

    diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
    panel = render_diff(diff)
    buffer = io.StringIO()
    Console(file=buffer, width=80).print(panel)
    output = buffer.getvalue()
    assert "old" in output
    assert "new" in output
    assert isinstance(Request, type)
