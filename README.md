# OpenTTD-LE

OpenTTD-LE is a benchmark scaffold for evaluating LLM and planning agents on
transport-logistics tasks inspired by OpenTTD. The current milestone establishes
the benchmark contracts: scenarios, structured observations, validated
macro-actions, agents, scoring, and reproducible artifacts.

The first backend is a deterministic logistics simulator used to exercise the
environment loop. The real OpenTTD backend is intentionally isolated behind the
same interface and is the next integration step.

## Quickstart

```bash
python -m pip install -e .
openttd-le list-scenarios
openttd-le eval --agent greedy --scenario coal_easy_001 --runs 1
openttd-le summarize runs
```

If the editable console script is not on `PATH`, use:

```bash
python -m openttd_le.cli eval --agent greedy --scenario coal_easy_001
```

Run an LLM agent with OpenAI:

```bash
$env:OPENAI_API_KEY="..."
openttd-le eval --agent openai --model gpt-5.5 --scenario coal_easy_001
```

Run through OpenRouter:

```bash
$env:OPENROUTER_API_KEY="..."
openttd-le eval --agent openrouter --model openai/gpt-5.5 --scenario coal_easy_001
```

## v0.1 Scope

- Fixed lab-play logistics scenarios.
- `reset -> observe -> step -> score` environment loop.
- Macro-actions: `build_route`, `add_vehicle`, `wait`, `take_loan`,
  `repay_loan`.
- Agents: random, greedy, OpenAI, OpenRouter.
- Artifacts: `summary.json`, `actions.jsonl`, `metrics.csv`,
  `final_state.json`, and `screenshots/final_map.svg`.
- Aggregation: `openttd-le summarize runs`.

## Backend Status

`toy` is the default backend and is useful for benchmark development. It is not
OpenTTD. The `openttd` backend performs real OpenTTD process integration: it
resolves `openttd.exe`, creates an isolated run directory, and can launch a
dedicated OpenTTD process.

```bash
openttd-le eval --backend toy --agent greedy --scenario coal_easy_001
openttd-le smoke-openttd --launch
```

## Visual Watch Mode

`watch-gpt` installs the bundled NoAI bridge into the OpenTTD user directory,
ensures OpenGFX is installed, writes an isolated launch config, and opens a
visible OpenTTD game. The current bridge executes a visible speedrun macro-plan:
choose two towns, label the objective/target, then rapidly build road bursts
near both selected towns with milestone signs.

```bash
openttd-le watch-gpt
```

If `OPENAI_API_KEY` is set, the command also records a GPT-5.5 plan artifact in
the run directory. This mode is useful for a fast visual smoke test; use
`play-gpt-live` when you want the model to choose each action during the run.

```bash
$env:OPENAI_API_KEY="..."
openttd-le watch-gpt --model gpt-5.5
```

## Live GPT Control

`play-gpt-live` starts a dedicated OpenTTD server with an OpenTTD-LE GameScript,
connects through the Admin Port, receives structured observations, asks the
model for a JSON action, sends the action back to the GameScript, and executes
the visible construction inside OpenTTD. A visible OpenTTD client is launched as
a spectator so you can watch the live server while the model acts.

```bash
$env:OPENAI_API_KEY="..."
openttd-le play-gpt-live --model gpt-5.5 --steps 4
```

For bridge testing without an API key:

```bash
openttd-le play-gpt-live --allow-heuristic --steps 2
```

Artifacts are written under `runs_live/<timestamp>_live_gpt/`, including
`live_trace.jsonl` with observations, model actions, and execution results.

## Coal Objective

`play-coal-live` runs the first real logistics objective. The GameScript exposes
coal-producing and coal-accepting industry pairs, GPT chooses a route, and the
bridge builds truck stops, a road path, a depot, road vehicles, and station
orders. Later steps wait for pickup/delivery and report objective metrics.

```bash
$env:OPENAI_API_KEY="..."
openttd-le play-coal-live --model gpt-5.5 --steps 6
```

For bridge testing without an API key:

```bash
openttd-le play-coal-live --allow-heuristic --steps 4
```

Artifacts are written under `runs_coal/<timestamp>_coal_objective/`, including
`coal_trace.jsonl`, `summary.json`, `launch.json`, and the run-specific
`openttd.cfg`.

## FIRS Excel Operations Loop

`play-firs-live` is the first workbook-driven FIRS loop:

```bash
$env:OPENAI_API_KEY="..."
openttd-le firs-init-workbook --out scenario.xlsx
openttd-le play-firs-live --workbook scenario.xlsx --model gpt-5.5 --record --out runs_firs
```

The command reads the workbook scenario/objectives, verifies that FIRS is
installed in the OpenTTD user directory, writes an isolated `openttd.cfg` with a
`[newgrf]` FIRS entry, launches the live GameScript bridge, asks the model for
macro-actions, and exports `report.xlsx` with actual routes, financials,
bottlenecks, and a scorecard.

The initial template uses the FIRS `basic_arctic` economy because it supports a
small Forest -> Sawmill chain. FIRS must be installed separately through
OpenTTD's Online Content UI before running:

```bash
openttd-le firs-init-workbook --config configs/firs_basic.toml --out scenario.xlsx
openttd-le play-firs-live --workbook scenario.xlsx --model gpt-5.5 --steps 10
openttd-le export-xlsx --run runs_firs/<timestamp>_firs_ops --workbook scenario.xlsx --out report.xlsx
```

For bridge testing without an API key:

```bash
openttd-le play-firs-live --workbook scenario.xlsx --allow-heuristic --steps 4
```

For FLE-style research runs, use the headless REPL benchmark command:

```bash
$env:OPENAI_API_KEY="..."
openttd-le firs-research-run --workbook scenario.xlsx --model gpt-5.5 --steps 32 --openttd-user-dir .openttd
openttd-le firs-research-run --workbook scenario.xlsx --task lab_supply_mine_short --steps 8 --openttd-user-dir .openttd
```

One research step is one generated Python program executed in a persistent
namespace. The exposed API is deliberately high-level: `observe()`,
`build_cargo_route()`, `add_vehicles()`, `wait_months()`,
`inspect_bottlenecks()`, `borrow_or_repay()`, typed `cargo_chains`,
`industries`, `finance`, and a small `Prototype` namespace. The benchmark writes
separate JSONL artifacts for programs, stdout/stderr, observations, actions, and
rewards so model comparisons do not depend on video recording. Research mode now
defaults to physical construction: `build_cargo_route()` attempts real OpenTTD
stations, roads, depots, and vehicles with `allow_virtual=False`, returning typed
failures when the bridge cannot build a continuous route.

Batch model/task comparisons use:

```bash
openttd-le firs-benchmark --workbook scenario.xlsx --tasks lab_supply_mine_short --models gpt-5.5 --repeats 3 --openttd-user-dir .openttd
```

Benchmark task definitions live in `scenarios/firs_benchmarks.json`.

Artifacts are written under `runs_firs/<timestamp>_firs_ops/`, including
`firs_trace.jsonl`, `summary.json`, `launch.json`, `report.xlsx`, the run-specific
`openttd.cfg`, and `gameplay.mp4` / `gameplay_8x.mp4` when `--record` is used
and `ffmpeg` is available.

## Test

```bash
python -m unittest discover -s tests
```
