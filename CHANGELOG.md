# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

Initial release. The CLI command is `qai`; the distribution on PyPI is
`qai-cli`; the repository is `local-coding-agent`.

### Added

- Local inference of Qwen2.5-Coder 1.5B (GGUF) through `llama-cpp-python`,
  with GGUF weights fetched from Hugging Face and cached on first run.
- Agentic file tools: `read_file`, `edit_file`, `write_file`, `list_dir`, `grep`.
- Workspace sandbox that rejects path traversal after symlink resolution, with
  `.git`, `node_modules`, `.venv` and model weights hidden by default.
- Write approval gate showing a unified diff, with `--yes` and `--read-only`
  overrides and a persistent "always allow" verdict.
- 32k context optimizer: reserved token budget, head-and-tail trimming of tool
  results, atomic tool-call blocks that cannot be orphaned, and auto-compaction
  of older turns with a deterministic extractive fallback.
- Context injection via `@path` mentions, `--attach` globs, and an automatic
  workspace manifest.
- One-shot and REPL modes, with `/help`, `/new`, `/context`, `/stats`, `/tools`,
  `/model` and `/read-only` commands.
- Layered configuration: defaults, `config.json`, `QAI_*` environment variables,
  then CLI flags.
- Optional `local` extra installing `llama-cpp-python`.
- CI on Linux, macOS and Windows; PyPI publishing via OIDC trusted publishing.
