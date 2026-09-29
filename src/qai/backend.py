"""Local model backend: GGUF download plus a llama.cpp chat/tool-call loop.

The backend is lazy-imported so that the rest of the CLI (config, workspace,
tool tests) works on machines without a llama.cpp build.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qai.config import Settings, model_cache_dir

_LLAMA_IMPORT_ERROR = (
    "llama-cpp-python is required to run the local model.\n"
    "Install it with:\n"
    "    pip install 'qai[local]'\n"
    "If you have no C++ toolchain, grab a prebuilt wheel from:\n"
    "    https://abetlen.github.io/llama-cpp-python/whl/cpu"
)


class ModelUnavailableError(RuntimeError):
    """The backend could not be loaded (missing build, missing weights, ...)."""


@dataclass(slots=True)
class Message:
    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = "call_0"


@dataclass(slots=True)
class Response:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


def ensure_model_file(settings: Settings, progress: bool = True) -> Path:
    """Return a local path to the GGUF weights, downloading them if needed."""
    target = model_cache_dir() / settings.model_file
    if target.is_file():
        return target

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ModelUnavailableError(
            "huggingface-hub is required to download model weights: pip install huggingface-hub"
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached = hf_hub_download(
            repo_id=settings.model_repo,
            filename=settings.model_file,
            cache_dir=str(model_cache_dir() / ".hf"),
        )
    except Exception as exc:  # network/auth/offline
        raise ModelUnavailableError(
            f"could not download {settings.model_repo}/{settings.model_file}: {exc}"
        ) from exc

    downloaded = Path(cached)
    if downloaded.resolve() != target.resolve():
        downloaded.replace(target)
    return target


class LlamaBackend:
    """Thin wrapper over ``llama_cpp.Llama`` exposing chat + tool calls."""

    def __init__(self, settings: Settings) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ModelUnavailableError(_LLAMA_IMPORT_ERROR) from exc

        self.settings = settings
        try:
            weights = ensure_model_file(settings)
        except ModelUnavailableError:
            raise

        kwargs: dict[str, Any] = {
            "model_path": str(weights),
            "n_ctx": settings.n_ctx,
            "n_gpu_layers": settings.n_gpu_layers,
            "chat_format": "chatml",
            "verbose": False,
        }
        if settings.n_threads:
            kwargs["n_threads"] = settings.n_threads
        try:
            self._llm = Llama(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on local build
            raise ModelUnavailableError(f"failed to load model: {exc}") from exc

    # -- prompting -------------------------------------------------------
    def stream(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> Iterator[str]:
        """Yield assistant text as it is produced."""
        if not self.settings.stream:
            yield self.complete(messages, tools).text
            return
        try:
            for chunk in self._llm.create_chat_completion(
                messages=[m.to_dict() for m in messages],
                tools=tools or None,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                max_tokens=self.settings.max_tokens,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    yield piece
        except Exception as exc:  # pragma: no cover - runtime inference failure
            raise ModelUnavailableError(f"inference failed: {exc}") from exc

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens_override: int | None = None,
    ) -> Response:
        """Run a single non-streaming completion and normalise tool calls."""
        try:
            raw = self._llm.create_chat_completion(
                messages=[m.to_dict() for m in messages],
                tools=tools or None,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                max_tokens=max_tokens_override or self.settings.max_tokens,
                stream=False,
            )
        except Exception as exc:  # pragma: no cover - runtime inference failure
            raise ModelUnavailableError(f"inference failed: {exc}") from exc

        return parse_response(raw)


_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def parse_response(raw: dict[str, Any]) -> Response:
    """Normalise a llama.cpp/OpenAI-shaped completion into a :class:`Response`.

    Small models frequently emit tool calls as a JSON code fence instead of a
    structured ``tool_calls`` field, so both shapes are accepted.
    """
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    finish_reason = choice.get("finish_reason") or "stop"

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(message.get("tool_calls") or []):
        function = raw_call.get("function") or {}
        name = function.get("name") or ""
        args = _as_dict(function.get("arguments"))
        if name:
            calls.append(
                ToolCall(
                    name=name,
                    arguments=args,
                    id=raw_call.get("id") or f"call_{index}",
                )
            )

    if not calls:
        calls = _parse_fenced_calls(text)
        if calls:
            finish_reason = "tool_calls"

    return Response(text=text, tool_calls=calls, finish_reason=finish_reason)


def _as_dict(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_fenced_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, match in enumerate(_FENCE_RE.finditer(text)):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        for item in payload if isinstance(payload, list) else [payload]:
            if not isinstance(item, dict):
                continue
            name = item.get("tool") or item.get("name")
            if not name:
                continue
            args = item.get("arguments") or item.get("args") or {}
            calls.append(
                ToolCall(
                    name=str(name),
                    arguments=_as_dict(args),
                    id=str(item.get("id") or f"call_{index}"),
                )
            )
    return calls
