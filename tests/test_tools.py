from __future__ import annotations

import pytest

from qai.backend import ToolCall, parse_response
from qai.tools import ToolRegistry, default_tools
from qai.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "a.py").write_text("def one():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def two():\n    return 2\n", encoding="utf-8")
    return Workspace(tmp_path)


@pytest.fixture
def registry(workspace):
    reg = ToolRegistry(workspace=workspace)
    reg.register_all(default_tools())
    return reg


def call(name, **args):
    return ToolCall(name=name, arguments=args)


def test_read_file(registry):
    result = registry.dispatch(call("read_file", path="a.py"))
    assert result.ok
    assert "def one()" in result.content
    assert "lines 1-2" in result.content


def test_read_file_line_range(registry):
    result = registry.dispatch(call("read_file", path="a.py", start_line=2, end_line=2))
    assert "return 1" in result.content
    assert "def one" not in result.content


def test_read_file_missing(registry):
    result = registry.dispatch(call("read_file", path="nope.py"))
    assert not result.ok
    assert "not found" in result.content


def test_read_file_escape_blocked(registry):
    result = registry.dispatch(call("read_file", path="../../etc/passwd"))
    assert not result.ok
    assert "escapes" in result.content


def test_grep(registry):
    result = registry.dispatch(call("grep", pattern=r"def \w+"))
    assert result.ok
    assert "a.py:1" in result.content
    assert "b.py:1" in result.content


def test_grep_bad_regex(registry):
    result = registry.dispatch(call("grep", pattern="([unclosed"))
    assert not result.ok
    assert "invalid" in result.content


def test_list_dir(registry):
    result = registry.dispatch(call("list_dir", path="."))
    assert "a.py" in result.content


def test_write_requires_approval(workspace):
    reg = ToolRegistry(workspace=workspace, approver=lambda req: "no")
    reg.register_all(default_tools())
    result = reg.dispatch(call("write_file", path="new.py", content="x = 1\n"))
    assert not result.ok
    assert "rejected" in result.content
    assert not (workspace.root / "new.py").exists()


def test_write_with_approval(workspace):
    reg = ToolRegistry(workspace=workspace, approver=lambda req: "yes")
    reg.register_all(default_tools())
    result = reg.dispatch(call("write_file", path="new.py", content="x = 1\n"))
    assert result.ok
    assert (workspace.root / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_always_stops_future_prompts(workspace):
    asked = []

    def approver(req):
        asked.append(req)
        return "always"

    reg = ToolRegistry(workspace=workspace, approver=approver)
    reg.register_all(default_tools())
    reg.dispatch(call("write_file", path="a.txt", content="1"))
    reg.dispatch(call("write_file", path="b.txt", content="2"))

    assert len(asked) == 1
    assert reg.auto_approve is True
    assert (workspace.root / "b.txt").exists()


def test_write_approval_offers_diff(workspace):
    seen = {}

    def approver(req):
        seen["summary"] = req.summary
        seen["diff"] = req.diff
        return "yes"

    reg = ToolRegistry(workspace=workspace, approver=approver)
    reg.register_all(default_tools())
    reg.dispatch(call("write_file", path="a.py", content="def one():\n    return 99\n"))
    assert "write a.py" in seen["summary"]
    assert "-    return 1" in seen["diff"]
    assert "+    return 99" in seen["diff"]


def test_read_only_blocks_writes(workspace):
    reg = ToolRegistry(workspace=workspace, read_only=True, auto_approve=True)
    reg.register_all(default_tools())
    result = reg.dispatch(call("write_file", path="x.py", content="x"))
    assert not result.ok
    assert "read-only" in result.content


def test_edit_file(registry, workspace):
    result = registry.dispatch(
        call("edit_file", path="a.py", old_string="return 1", new_string="return 42")
    )
    assert result.ok
    assert "return 42" in (workspace.root / "a.py").read_text(encoding="utf-8")


def test_edit_file_ambiguous(workspace):
    (workspace.root / "c.py").write_text("x\nx\n", encoding="utf-8")
    reg = ToolRegistry(workspace=workspace, auto_approve=True)
    reg.register_all(default_tools())
    result = reg.dispatch(call("edit_file", path="c.py", old_string="x", new_string="y"))
    assert not result.ok
    assert "2 times" in result.content


def test_edit_file_replace_all(workspace):
    (workspace.root / "c.py").write_text("x\nx\n", encoding="utf-8")
    reg = ToolRegistry(workspace=workspace, auto_approve=True)
    reg.register_all(default_tools())
    result = reg.dispatch(
        call("edit_file", path="c.py", old_string="x", new_string="y", replace_all=True)
    )
    assert result.ok
    assert (workspace.root / "c.py").read_text(encoding="utf-8") == "y\ny\n"


def test_edit_file_requires_read_first_not_enforced_but_safe(workspace):
    reg = ToolRegistry(workspace=workspace, auto_approve=True)
    reg.register_all(default_tools())
    result = reg.dispatch(
        call("edit_file", path="a.py", old_string="does not exist", new_string="y")
    )
    assert not result.ok
    assert "not found" in result.content


def test_unknown_tool(registry):
    result = registry.dispatch(call("teleport"))
    assert not result.ok
    assert "unknown tool" in result.content


def test_schema_shape(registry):
    schema = registry.schema()
    assert {entry["function"]["name"] for entry in schema} == {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
    }
    write = next(e for e in schema if e["function"]["name"] == "write_file")
    assert write["function"]["parameters"]["required"] == ["path", "content"]


def test_parse_native_tool_calls():
    raw = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.py"}',
                            },
                        }
                    ],
                },
            }
        ]
    }
    response = parse_response(raw)
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "a.py"}


def test_parse_fenced_tool_call():
    raw = {
        "choices": [
            {
                "message": {
                    "content": (
                        'Let me look.\n```json\n'
                        '{"tool": "grep", "arguments": {"pattern": "x"}}\n```'
                    )
                }
            }
        ]
    }
    response = parse_response(raw)
    assert response.tool_calls[0].name == "grep"
    assert response.finish_reason == "tool_calls"
