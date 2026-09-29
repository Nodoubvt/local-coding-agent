"""Configuration loading: defaults -> user config file -> env vars -> CLI flags."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir

APP_NAME = "qai"

DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

DEFAULT_SYSTEM_PROMPT = """You are qai, a concise terminal coding assistant running \
inside a user's workspace. You can read and edit files with tools.

Rules:
- Read a file before editing it. Never guess file contents.
- Prefer the smallest change that solves the request; match existing style.
- Use relative paths and do not touch files outside the workspace.
- When you change code, state briefly what you changed and why.
- Be terse. No preamble, no summaries you were not asked for.
"""


@dataclass(slots=True)
class Settings:
    """Runtime configuration for one CLI session."""

    model_repo: str = DEFAULT_MODEL_REPO
    model_file: str = DEFAULT_MODEL_FILE
    n_ctx: int = 32_768
    n_gpu_layers: int = -1
    n_threads: int | None = None
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 4_096
    max_iterations: int = 12
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    auto_approve: bool = False
    stream: bool = True
    read_only: bool = False
    # context management
    auto_compact: bool = True
    compact_threshold: float = 0.70
    max_tool_result_chars: int = 8_000
    keep_recent_blocks: int = 6
    extra_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_dir() -> Path:
    override = os.environ.get("QAI_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir(APP_NAME, appauthor=False, roaming=True))


def config_path() -> Path:
    return config_dir() / "config.json"


def model_cache_dir() -> Path:
    override = os.environ.get("QAI_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_cache_dir(APP_NAME, appauthor=False)) / "models"


_SETTINGS_FIELDS = {f.name: f for f in fields(Settings)}
_ENV_MAP = {
    "QAI_MODEL_REPO": "model_repo",
    "QAI_MODEL_FILE": "model_file",
    "QAI_N_CTX": "n_ctx",
    "QAI_N_GPU_LAYERS": "n_gpu_layers",
    "QAI_N_THREADS": "n_threads",
    "QAI_TEMPERATURE": "temperature",
    "QAI_TOP_P": "top_p",
    "QAI_MAX_TOKENS": "max_tokens",
    "QAI_MAX_ITERATIONS": "max_iterations",
    "QAI_SYSTEM_PROMPT": "system_prompt",
    "QAI_AUTO_APPROVE": "auto_approve",
    "QAI_STREAM": "stream",
    "QAI_READ_ONLY": "read_only",
    "QAI_AUTO_COMPACT": "auto_compact",
    "QAI_COMPACT_THRESHOLD": "compact_threshold",
    "QAI_MAX_TOOL_RESULT_CHARS": "max_tool_result_chars",
    "QAI_KEEP_RECENT_BLOCKS": "keep_recent_blocks",
}

_INT_FIELDS = {
    "n_ctx",
    "n_gpu_layers",
    "n_threads",
    "max_tokens",
    "max_iterations",
    "max_tool_result_chars",
    "keep_recent_blocks",
}
_FLOAT_FIELDS = {"temperature", "top_p", "compact_threshold"}
_BOOL_FIELDS = {"auto_approve", "stream", "read_only", "auto_compact"}


def _coerce(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name in _BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if name in _INT_FIELDS:
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    return value


def load_settings(overrides: dict[str, Any] | None = None) -> Settings:
    """Merge config file, env vars and explicit overrides (highest priority last)."""
    values: dict[str, Any] = {}

    path = config_path()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key in _SETTINGS_FIELDS:
                    values[key] = value

    for env_key, name in _ENV_MAP.items():
        if env_key in os.environ:
            values[name] = os.environ[env_key]

    for key, value in (overrides or {}).items():
        if value is not None and key in _SETTINGS_FIELDS:
            values[key] = value

    settings = Settings()
    for key, value in values.items():
        coerced = _coerce(key, value)
        if coerced is None and key not in ("system_prompt",):
            continue
        setattr(settings, key, coerced)
    return settings


def save_settings(settings: Settings) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
