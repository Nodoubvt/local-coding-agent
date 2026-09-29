from __future__ import annotations

import pytest

from qai.agent import Agent
from qai.backend import Message, Response, ToolCall
from qai.config import Settings
from qai.context import collect_context, extract_mentions
from qai.tools import ToolRegistry, default_tools
from qai.workspace import Workspace


class FakeBackend:
    """Replays a scripted list of responses and records the prompts it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[Message]] = []
        self.settings = Settings(stream=False)

    def complete(self, messages, tools=None, max_tokens_override=None):
        self.calls.append(list(messages))
        if not self.responses:
            return Response(text="done")
        return self.responses.pop(0)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    return Workspace(tmp_path)


def test_extract_mentions():
    cleaned, mentions = extract_mentions("fix @src/app.py and @tests/test_x.py please")
    assert mentions == ["src/app.py", "tests/test_x.py"]
    assert "@" not in cleaned
    assert "fix" in cleaned


def test_collect_context_includes_manifest_and_file(workspace):
    block = collect_context(workspace, ["a.py"])
    assert "<workspace" in block
    assert 'path="a.py"' in block
    assert "print(1)" in block


def test_collect_context_without_manifest(workspace):
    block = collect_context(workspace, ["a.py"], include_manifest=False)
    assert "<workspace" not in block
    assert 'path="a.py"' in block


def test_agent_runs_tool_then_answers(workspace):
    tool_response = Response(
        text='```json\n{"tool": "read_file", "arguments": {"path": "a.py"}}\n```',
        tool_calls=[ToolCall(name="read_file", arguments={"path": "a.py"})],
    )
    backend = FakeBackend([tool_response, Response(text="The file prints 1.")])
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    agent = Agent(backend=backend, registry=registry, system_prompt="sys")
    result = agent.run([Message(role="user", content="what does a.py do?")])

    assert result.text == "The file prints 1."
    assert len(backend.calls) == 2
    tool_messages = [m for m in backend.calls[1] if m.role == "tool"]
    assert tool_messages and "print(1)" in tool_messages[0].content


def test_agent_stops_at_iteration_limit(workspace):
    loop = Response(tool_calls=[ToolCall(name="list_dir", arguments={})])
    backend = FakeBackend([loop] * 10)
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    agent = Agent(backend=backend, registry=registry, max_iterations=3, system_prompt="sys")
    result = agent.run([Message(role="user", content="loop forever")])

    assert "step limit" in result.text
    assert len(backend.calls) == 3


def test_agent_respects_rejection(workspace):
    tool_response = Response(
        tool_calls=[ToolCall(name="write_file", arguments={"path": "z.py", "content": "z"})]
    )
    backend = FakeBackend([tool_response, Response(text="ok")])
    registry = ToolRegistry(workspace=workspace, approver=lambda req: "no")
    registry.register_all(default_tools())

    agent = Agent(backend=backend, registry=registry, system_prompt="sys")
    agent.run([Message(role="user", content="write z")])

    tool_messages = [m for m in backend.calls[1] if m.role == "tool"]
    assert "rejected" in tool_messages[0].content
    assert not (workspace.root / "z.py").exists()


def test_agent_injects_system_prompt_and_tools(workspace):
    backend = FakeBackend([Response(text="hello")])
    registry = ToolRegistry(workspace=workspace, auto_approve=True)
    registry.register_all(default_tools())

    agent = Agent(backend=backend, registry=registry, system_prompt="sys")
    agent.run([Message(role="user", content="hi")])

    first = backend.calls[0][0]
    assert first.role == "system"
    assert "sys" in first.content
    assert "```json" in first.content
