<div align="center">

![CI](https://github.com/Nodoubvt/local-coding-agent/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-2563eb.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)
![Model: Qwen2.5-Coder 1.5B](https://img.shields.io/badge/model-Qwen2.5--Coder--1.5B-f97316.svg)
![Inference: local / offline](https://img.shields.io/badge/inference-100%25%20local-16a34a.svg)
![No API key](https://img.shields.io/badge/API%20keys-none-16a34a.svg)
![Stars](https://img.shields.io/github/stars/Nodoubvt/local-coding-agent?style=social)

</div>

# local-coding-agent

**An offline AI coding assistant that runs entirely on your own machine.**
`qai` is a terminal coding agent powered by **Qwen2.5-Coder 1.5B** running on
[llama.cpp](https://github.com/ggml-org/llama.cpp) — no cloud, no API keys, no
telemetry, no data leaving your laptop.

It is deliberately small: about **1,700 lines** of readable Python you can finish
in one sitting, built to be extended rather than depended on.

> **CLI command:** `qai` · **PyPI package:** `qai-cli` · **Repository:**
> `local-coding-agent` · **License:** MIT

```
qai> add a docstring to every function in utils.py
  tool read_file(path=utils.py)
ok  read_file
  tool edit_file(path=utils.py, old_string=..., new_string=...)
  Approve edit utils.py
  ▸ 12 lines changed
```

---

## Why this exists

Most "AI coding agent" tools assume a 70B model, a GPU, and a subscription.
This one assumes nothing:

- **Runs a 1.5B model on a laptop CPU.** No CUDA, no 24 GB of VRAM, no
  electricity bill. It is a *small* model, and the tooling around it is built to
  make a small model useful.
- **~1.1 GB download.** Q4_K_M GGUF weights, cached after the first run.
- **1,700 lines, fully readable.** The whole agent — sandbox, tools, approval
  gate, context optimizer — is short enough to read, audit and fork.
- **MIT, no strings.** The safety model is code you can inspect, not a policy
  page.

## Features

- **Local inference** — GGUF weights from Hugging Face, run through
  `llama-cpp-python`. CPU-only works out of the box; GPU offload is on by default
  when a backend exists.
- **Agentic file tools** — `read_file`, `edit_file`, `write_file`, `list_dir`,
  `grep`, all sandboxed to the workspace root.
- **Approval gate** — every write shows a unified diff and waits for you. `--yes`
  skips it, `--read-only` disables writes entirely.
- **32k context with a real optimizer** — tool results and file dumps are trimmed
  head-and-tail, then older turns are auto-compacted into a summary before the
  buffer fills. Tool calls are never orphaned from their results.
- **Context injection** — `@path` mentions, `--attach` globs, and an automatic
  workspace manifest so a 1.5B model knows what files exist.
- **Two modes** — one-shot `qai "..."` for scripting, or a REPL with `@file`
  completion, history and slash commands.

## Requirements

- Python 3.10 or newer
- ~2 GB free disk for weights, ~2 GB RAM at inference
- A C++ toolchain, *or* a prebuilt `llama-cpp-python` wheel (below)

## Install

### Fastest — no install, no virtualenv

```bash
uvx qai-cli              # run it instantly
```

### Global install, still isolated

```bash
uv tool install qai-cli  # or: pipx install qai-cli
qai "what does main() do?"
```

> **Local inference needs `llama-cpp-python`**, a compiled C++ extension. If you
> have no C++ toolchain, use the prebuilt CPU wheel:
>
> ```bash
> pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
> ```
>
> Then run `uvx --with llama-cpp-python qai-cli` to try it in one shot.

### From source

```bash
git clone https://github.com/Nodoubvt/local-coding-agent
cd local-coding-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[local]"         # [local] adds llama-cpp-python
```

Inspect resolved settings, or fetch weights ahead of time:

```bash
qai --print-config
python -c "from qai.config import load_settings; from qai.backend import ensure_model_file; print(ensure_model_file(load_settings()))"
```

## Usage

```bash
qai                                  # interactive session
qai "why does main() hang here?"     # one-shot
qai "refactor this" -a src/api.py    # attach a file
qai "find every TODO" --no-tools     # pure chat, no filesystem access

qai --model qwen2.5-coder-1.5b-instruct-q8_0.gguf   # higher-quality quant
qai --gpu-layers 0 --threads 8       # force CPU, pin thread count
qai --read-only                      # refuse every write
qai --compact-at 0.5                 # summarize earlier, at 50% of budget
qai --no-compact                     # disable the compactor
```

### REPL commands

| command | effect |
| --- | --- |
| `/help` | show commands |
| `/new` | clear conversation history |
| `/context [path]` | re-send the manifest, or attach specific paths |
| `/stats` | context usage, budget, compaction count |
| `/tools` | list available tools |
| `/model` | show active model and context size |
| `/read-only on\|off` | toggle write access |
| `/exit` | quit (ctrl-d also works) |

## Configuration

Settings merge in this order: **defaults → `config.json` → env vars → CLI flags**.

```bash
python -c "from qai.config import Settings, save_settings; save_settings(Settings(n_ctx=32768))"
```

| setting | default | meaning |
| --- | --- | --- |
| `n_ctx` | `32768` | context window |
| `max_tokens` | `4096` | reply ceiling, also reserved from the budget |
| `compact_threshold` | `0.70` | auto-compact above this context ratio |
| `max_tool_result_chars` | `8000` | per-result trim limit |
| `keep_recent_blocks` | `6` | turns kept verbatim through compaction |
| `auto_compact` | `true` | enable the compactor |
| `n_gpu_layers` | `-1` | `-1` offload all, `0` CPU only |
| `temperature` | `0.2` | sampling temperature |
| `auto_approve` | `false` | skip write approval |

Env equivalents: `QAI_N_CTX`, `QAI_TEMPERATURE`, `QAI_AUTO_COMPACT`,
`QAI_MODEL_DIR`, `QAI_CONFIG_DIR`, and the rest follow `QAI_` + the setting name.

## How the context optimizer works

A 1.5B model with a 32k window still hits the wall fast, because a single
`read_file` of a large file can eat a quarter of it. Three layers keep a long
session degrading instead of failing:

1. **Budget.** `budget = n_ctx - max_tokens - tool_schema_reserve` — 27,472
   tokens at defaults. Everything is measured with a deliberately pessimistic
   chars-per-token estimate, so the agent under-fills rather than overflows.
2. **Trim.** Oversized tool results and `<context>` blocks are cut head-and-tail,
   so the end of an error message survives. Assistant messages carrying
   `tool_calls` are grouped with their results into a single atomic block, so
   trimming can never leave an orphan that breaks the chat template.
3. **Compact.** Past `compact_threshold`, older turns are summarized — by the
   model when it is available, otherwise by a deterministic extractive fallback —
   into a single `<summary>` block, with the last `keep_recent_blocks` turns kept
   verbatim.

Compaction runs between tool rounds, never mid-flight, so the model always sees a
consistent transcript, and the compacted transcript is carried into the next turn
so a long REPL session stays bounded instead of being re-trimmed from scratch
every time. `/stats` shows live usage.

## Safety

- All tool paths resolve through `Workspace`, which refuses anything escaping the
  workspace root after symlink resolution.
- `.git`, `node_modules`, `.venv` and model weights are hidden from listing,
  search and context injection by default; add your own with `--ignore`.
- Writes require explicit approval with a diff shown first.
- `--read-only` disables every mutating tool at the registry level, not just in
  the prompt.

## Development

```bash
pip install -e ".[dev]"
pytest              # 70 tests, no model or network required
ruff check .
mypy src
```

Tests run against a scripted fake backend, so nothing downloads 1 GB. Coverage
includes sandbox escapes, the approval gate, tool-call pairing across trims, and
a simulated 32k session that reads a 20k-line file fifteen times.

## Releasing

Push a `v*` tag. The workflow verifies the tag matches the `pyproject.toml`
version, builds the sdist and wheel, checks metadata, smoke-tests the wheel, and
publishes to PyPI via OIDC trusted publishing — no API token is stored in the
repository.

```bash
# 1. bump version in pyproject.toml + CHANGELOG.md, commit
# 2. tag and push
git tag v0.1.0 && git push origin v0.1.0
```

## License

MIT — see [LICENSE](LICENSE).

---

## Beyond the terminal

<p align="center">
  <img src="docs/images/vibe.png" alt="Agentic Studio in Vibe Mode: a multi-pane IDE with a VS-style code editor, a live tool-call log showing grep_files and execute_shell, an agent conversation panel, and L1-L4 context cache gauges along the bottom" width="100%">
</p>

<p align="center"><em>Vibe Mode — the agent snapshots the workspace, then runs
unattended with the sandbox still hard-caged. Note the live tool log and the
L1–L4 context gauges along the bottom edge.</em></p>

<p align="center">
  <img src="docs/images/ss-ctx1.png" alt="The Context window in Agentic Studio: a model picker, a 125k token slider with 16k through 1M presets, and below it the Token Context Heap heatmap colouring each cluster as free space, pinned, reading, writing, bloated or defragmented" width="100%">
</p>

<p align="center"><em>The Context Engine — a per-model window from 16k to 1M,
with the Token Context Heap showing exactly which clusters are pinned, bloating
or still free.</em></p>

`qai` is the terminal. If you want the graphical version of the same idea —
**5 coordinated agents, an inspectable context engine, embedded PHP/MariaDB/WebGPU
emulators and one-click rollback** — that is
**[Devhead Agentic Studio](https://dev-head.com/products/astudio.html)**, a
separate commercial product built by the same author.

The two are deliberately different tools: `qai` is free, MIT, and 1,700 lines of
Python you can audit today; Agentic Studio is a Windows desktop IDE. The CLI
stays free and MIT-licensed either way — nothing here is a trial, and nothing
here phones home.

*(Screenshots above are from the author's own product page and remain the
property of their respective owner.)*
