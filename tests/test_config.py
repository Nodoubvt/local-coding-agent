from __future__ import annotations

import json

from qai.config import Settings, load_settings, save_settings


def test_defaults():
    settings = load_settings()
    assert settings.model_repo == "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
    assert settings.n_ctx == 32768


def test_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setenv("QAI_CONFIG_DIR", str(tmp_path))
    save_settings(Settings(n_ctx=2048, temperature=0.9))
    settings = load_settings({"n_ctx": 4096})
    assert settings.n_ctx == 4096
    assert settings.temperature == 0.9


def test_env_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("QAI_CONFIG_DIR", str(tmp_path))
    save_settings(Settings(n_ctx=2048))
    monkeypatch.setenv("QAI_N_CTX", "1024")
    assert load_settings().n_ctx == 1024


def test_bool_and_float_coercion(tmp_path, monkeypatch):
    monkeypatch.setenv("QAI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("QAI_AUTO_APPROVE", "yes")
    monkeypatch.setenv("QAI_TEMPERATURE", "0.75")
    settings = load_settings()
    assert settings.auto_approve is True
    assert settings.temperature == 0.75


def test_none_override_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("QAI_CONFIG_DIR", str(tmp_path))
    assert load_settings({"n_ctx": None}).n_ctx == 32768


def test_saved_config_is_valid_json(tmp_path, monkeypatch):
    monkeypatch.setenv("QAI_CONFIG_DIR", str(tmp_path))
    path = save_settings(Settings())
    assert json.loads(path.read_text(encoding="utf-8"))["n_ctx"] == 32768


def test_corrupt_config_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("QAI_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    assert load_settings().n_ctx == 32768
