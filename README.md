# local-coding-agent

A tiny terminal coding agent that runs **Qwen2.5-Coder 1.5B** entirely on your
machine, via [llama.cpp](https://github.com/ggml-org/llama.cpp). No API keys, no
telemetry, no data leaving your laptop.

It is a small, readable codebase on purpose — around 1,700 lines you can read in
one sitting — built to be extended rather than depended on. The CLI command is
`qai`; the project is `local-coding-agent`.

```
qai> add a docstring to every function in utils.py
  read_file(utils.py)
ok read_file
  edit_file(utils.py)
  Approve edit utils.py
```

---

## Features

- **Local inference.** GGUF weights are downloaded once from Hugging Face and run
  through `llama-cpp-python`. CPU-only works out of the box; GPU offload is on by
  default when a backend is available.
- **Agentic file tools.** `read_file`, `edit_file`, `write_file`, `list_dir` and
  `grep`, all sandboxed to the workspace root.
- **Approval gate.** Every write shows a unified diff and waits for you. `--yes`
  skips it, `--read-only` disables writes entirely.
- **32k context with an optimizer.** Tool results and file dumps are trimmed
  head-and-tail, then older turns are auto-compacted into a summary before the
  buffer fills. Never orphans a tool call from its result.
- **Context injection.** `@path` mentions, `--attach` globs, and an automatic
  workspace manifest so a 1.5B model knows what files exist.
- **Two modes.** One-shot `qai "..."` for scripting, or a REPL with
  `@file` completion, history and slash commands.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/Nodoubvt/local-coding-agent
cd local-coding-agent

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[local]"
```

`[local]` pulls in `llama-cpp-python`, which compiles a small C++ extension. If
you have no C++ toolchain, grab a prebuilt wheel instead:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

The first run downloads ~1.1 GB of Q4_K_M weights and caches them. To fetch them
ahead of time, or to run without the agent tools:

```bash
qai --print-config          # inspect resolved settings
python -c "from qai.config import load_settings, save_settings; from qai.backend import ensure_model_file; print(ensure_model_file(load_settings()))"
```

## Usage

```bash
qai                                  # interactive session
qai "why does main() hang here?"     # one-shot
qai "refactor this" -a src/api.py    # attach a file
qai "find every TODO" --no-tools     # pure chat, no filesystem access

qai --model qwen2.5-coder-1.5b-instruct-q8_0.gguf
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
Write the file with:

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

Three layers, in order, so a long session degrades instead of failing:

1. **Budget.** `budget = n_ctx - max_tokens - tool_schema_reserve` (27,472 tokens
   at defaults). Everything is measured with a deliberately pessimistic
   chars-per-token estimate so you under-fill rather than overflow.
2. **Trim.** Oversized tool results and `<context>` blocks are cut
   head-and-tail, so the end of an error message survives. Assistant messages
   carrying `tool_calls` are grouped with their results into a single atomic
   block, so trimming can never leave an orphan that breaks the chat template.
3. **Compact.** Past `compact_threshold`, older turns are summarized — by the
   model when it is available, otherwise by a deterministic extractive fallback
   — into a single `<summary>` block, with the last `keep_recent_blocks` turns
   kept verbatim.

Compaction runs between tool rounds, never mid-flight, so the model always sees a
consistent transcript, and the compacted transcript is carried into the next turn
so a long REPL session stays bounded instead of being re-trimmed from scratch
every time. `/stats` shows live usage; a one-line notice appears whenever a
compaction happens.

## Safety

- All tool paths resolve through `Workspace`, which refuses anything that escapes
  the workspace root after symlink resolution.
- `.git`, `node_modules`, `.venv`, model weights and friends are hidden from
  listing, search and context injection by default; pass `--ignore` patterns to
  add your own.
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

The test suite uses a scripted fake backend, so nothing downloads 1 GB to run it.
Coverage includes sandbox escapes, the approval gate, tool-call pairing across
trims, and a simulated 32k session that reads a 20k-line file fifteen times. To
try a real model, install `[local]` and run `qai` in a scratch directory.

## License

MIT — see [LICENSE](LICENSE).

---

### Looking for a graphical environment?

If you like running local models like Qwen2.5-Coder in the terminal but want
multi-agent coordination, long-horizon context management and a visual
workspace, **[Devhead Agentic Studio](https://dev-head.com/products/astudio.html)**
is the graphical evolution of this tool, built by the same author. The CLI above
stays free and MIT-licensed either way.
