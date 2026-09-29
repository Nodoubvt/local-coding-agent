from __future__ import annotations

import os

import pytest

from qai.workspace import SandboxError, Workspace


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    (tmp_path / "model.gguf").write_text("bin", encoding="utf-8")
    return Workspace(tmp_path)


def test_resolve_relative(workspace):
    assert workspace.resolve("src/app.py") == workspace.root / "src" / "app.py"


def test_resolve_rejects_escape(workspace):
    with pytest.raises(SandboxError):
        workspace.resolve("../outside.txt")


def test_resolve_rejects_absolute_outside(workspace):
    # Use a genuinely absolute path for the running platform: on POSIX a
    # Windows-style "C:/..." string is a *relative* path and legitimately
    # resolves inside the root, so a hardcoded drive path would not raise.
    with pytest.raises(SandboxError):
        workspace.resolve(os.path.abspath(os.sep))


def test_resolve_rejects_sibling_directory(workspace):
    sibling = workspace.root.parent / "elsewhere"
    with pytest.raises(SandboxError):
        workspace.resolve(sibling)


def test_resolve_rejects_empty(workspace):
    with pytest.raises(SandboxError):
        workspace.resolve("   ")


def test_ignores_default_patterns(workspace):
    assert workspace.is_ignored(workspace.root / "node_modules")
    assert workspace.is_ignored(workspace.root / "model.gguf")
    assert not workspace.is_ignored(workspace.root / "src" / "app.py")


def test_walk_skips_ignored(workspace):
    names = {workspace.relative(p) for p in workspace.walk()}
    assert names == {"src/app.py"}


def test_relative_uses_posix(workspace):
    assert workspace.relative(workspace.root / "src" / "app.py") == "src/app.py"
