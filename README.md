# OpenTTD-LE

OpenTTD-LE is a real OpenTTD/FIRS environment layer for evaluating LLM and
planning agents on transport-logistics tasks. The primary path launches
OpenTTD, drives the bundled GameScript/Admin Port bridge, executes physical
macro-actions in the game, and writes researcher artifacts: actions,
observations, rewards, reports, replay manifests, and optional gameplay video.

The core environment follows the Farama/Gymnasium separation: the environment
owns `reset -> step -> reward -> done`, while GPT-5.5 is only one replaceable
baseline agent.

Toy mode exists only as a local mock backend for unit tests, CI, and fast API
debugging. It is not the research environment.

## OpenTTD Research Quickstart

Install OpenTTD, install FIRS through OpenTTD's Online Content UI, then run:

```bash
python -m pip install -e .
openttd-le firs-init-workbook --config configs/firs_basic.toml --out scenario.xlsx
set "OPENTTD_USER_DIR=%CD%\.openttd"
openttd-le smoke-openttd --firs --openttd-user-dir .openttd
openttd-le list-openttd-scenarios
$env:OPENAI_API_KEY="..."
openttd-le eval --scenario lab_raw_to_processor --model gpt-5.5 --workbook scenario.xlsx --openttd-user-dir .openttd --out runs_openttd
```

For bridge testing without an API key:

```bash
openttd-le eval --scenario lab_raw_to_processor --agent heuristic --workbook scenario.xlsx --openttd-user-dir .openttd --out runs_openttd
```

Use the native environment directly:

```python
from openttd_le.envs import OpenTTDFIRSEnv

env = OpenTTDFIRSEnv(
    workbook="scenario.xlsx",
    task_id="lab_raw_to_processor",
    openttd_user_dir=".openttd",
    seed=1,
)
obs, info = env.reset()
action = info["candidate_actions"][0]["action"]
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

The real environment accepts macro-actions:

```python
{"type": "build_cargo_route", "source_id": 29, "destination_id": 12, "cargo_id": 2, "vehicles": 5}
{"type": "wait_months", "months": 1}
{"type": "add_vehicles", "route_id": "route_1", "count": 1}
{"type": "inspect_bottlenecks"}
```

`candidate_actions` are suggestions exposed in `obs` and `info`; the environment
does not choose them for the agent. The OpenAI/GPT path is a baseline agent
that consumes the same observations and returns one macro-action at a time.

Use the optional Gymnasium adapter when an RL loop needs fixed spaces:

```python
from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv

env = OpenTTDFIRSGymEnv(
    workbook="scenario.xlsx",
    task_id="lab_raw_to_processor",
    openttd_user_dir=".openttd",
    max_candidates=24,
)
obs, info = env.reset(seed=1)
action_mask = info["action_mask"]
obs, reward, terminated, truncated, info = env.step(0)
env.close()
```

The adapter exposes `Discrete(max_candidates)` over the current macro-action
frontier. Rich OpenTTD state remains available as `info["native_observation"]`;
`info["candidate_actions"]` and `info["action_mask"]` are the standard control
surface for masked-action RL algorithms.

The registered Gymnasium IDs are:

- `OpenTTD-FIRS-Lab-v0`: real OpenTTD/FIRS, launches OpenTTD on `reset()`.
- `OpenTTD-FIRS-Deterministic-v0`: real OpenTTD/FIRS with deterministic API observations and normalized info.
- `OpenTTDLE-Toy-v0`: mock backend for CI and interface debugging.

```python
import gymnasium as gym
import openttd_le.adapters.gymnasium

env = gym.make(
    "OpenTTD-FIRS-Lab-v0",
    workbook="scenario.xlsx",
    openttd_user_dir=".openttd",
    max_candidates=24,
)
```

Validate the adapter contract:

```bash
openttd-le check-gym-env --backend toy
openttd-le check-gym-env --backend openttd --scenario lab_raw_to_processor --workbook scenario.xlsx --openttd-user-dir .openttd
openttd-le check-gym-env --backend openttd --deterministic --scenario lab_raw_to_processor --workbook scenario.xlsx --openttd-user-dir .openttd
```

The real FIRS Gym observation is a fixed-shape `spaces.Dict`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `tick` | scalar `float32` | OpenTTD game tick |
| `bank_balance` | scalar `float32` | company cash balance |
| `route_count` | scalar `float32` | registered physical cargo routes |
| `delivered_routes` | scalar `float32` | routes with at least one delivery |
| `cargo_delivered` | scalar `float32` | total cargo units delivered |
| `route_profit` | scalar `float32` | summed route vehicle profit |
| `candidate_production` | `float32[max_candidates]` | production estimate for each candidate route action |
| `action_mask` | `int8[max_candidates]` | valid candidate indexes for masked-action algorithms |

The wrapper also implements `env.action_masks()` for libraries such as
MaskablePPO. Full structured state stays in `info["native_observation"]`, not
the policy tensor.

Parallel rollout helpers are available for small worker counts:

```python
from openttd_le.adapters.gymnasium import make_firs_vector

envs = make_firs_vector(
    2,
    workbook="scenario.xlsx",
    task_id="lab_raw_to_processor",
    openttd_user_dir=".openttd",
)
```

Each worker launches its own OpenTTD process and gets unique ephemeral network
ports. This is suitable for low-parallelism research experiments; large-scale
parallel RL needs enough CPU/RAM for multiple OpenTTD instances.

Non-GPT Gym baselines:

```bash
openttd-le benchmark-gym --workbook scenario.xlsx --scenario lab_raw_to_processor --openttd-user-dir .openttd --agents masked_random,first_valid,highest_production,shortest_route --seeds 1,2,3
python examples/random_gym_rollout.py --workbook scenario.xlsx --openttd-user-dir .openttd
python examples/train_maskable_ppo.py --workbook scenario.xlsx --openttd-user-dir .openttd --timesteps 64
python examples/train_dqn_macro.py --workbook scenario.xlsx --openttd-user-dir .openttd --timesteps 128
```

Researcher training/eval harness:

```bash
openttd-le train-rl-baselines --workbook scenario.xlsx --scenario lab_raw_to_processor --openttd-user-dir .openttd --algorithms scripted:masked_random,scripted:first_valid --seeds 1,2 --out runs_rl
openttd-le train-rl-baselines --workbook scenario.xlsx --scenario lab_raw_to_processor --openttd-user-dir .openttd --algorithms dqn --total-timesteps 128 --eval-interval 64 --out runs_rl_dqn
```

Install `openttd-le[rl]` or `stable-baselines3 sb3-contrib` before running
`dqn` or `maskable_ppo`. The scripted algorithms are dependency-free controls.
The harness writes `rl_training_report.json`, `benchmark_report.md`, CSV tables,
and SVG learning curves.

Determinism check:

```bash
openttd-le determinism-check --workbook scenario.xlsx --scenario lab_raw_to_processor --openttd-user-dir .openttd --agent first_valid --seed 1 --repeats 3 --max-steps 3
```

This command reruns the same baseline action sequence against real OpenTTD/FIRS
and compares normalized observations, rewards, termination flags, and Gym info.
It writes `determinism_report.json` and exits non-zero on the first mismatch.
The deterministic Gym mode removes volatile runtime fields such as ports, PIDs,
run directories, raw game ticks, viewport scrolls, and vehicle pixel positions
from the API comparison surface. The raw simulator artifacts remain in each run
directory for audit and replay.

Benchmark validity pack:

```bash
openttd-le benchmark-validity-pack --workbook scenario.xlsx --openttd-user-dir .openttd --out runs_validity
```

The validity pack is the pre-release research gate. It loads
`scenarios/firs_validity_suite.json`, writes `suite_manifest.json`, then runs:

- deterministic repeated traces for each task/seed
- non-GPT Gym baselines over the official task suite
- throughput measurements: reset time, step time, transitions/hour
- physical route-builder reliability for each seed/economy pair

For a quick smoke run during development:

```bash
openttd-le benchmark-validity-pack --workbook scenario.xlsx --openttd-user-dir .openttd --tasks lab_raw_to_processor --agents first_valid --seeds 1 --determinism-repeats 2 --determinism-max-steps 2 --baseline-max-steps 2 --throughput-steps 2 --skip-route-builder --out runs_validity_smoke
```

The full report is written to `validity_report.json`. This is the artifact to
inspect before claiming the environment is suitable for serious RL experiments.
It also writes `benchmark_report.md`, CSV tables under `tables/`, and SVG curves
under `curves/` when training data is supplied.

To merge validity and training reports into one artifact:

```bash
openttd-le build-benchmark-report --validity-report runs_validity/validity_report.json --training-report runs_rl/rl_training_report.json --route-builder-report runs_route_builder/<run>/summary.json --out runs_research_report
```

For reproducibility, each real run records `seed`, `economy`, FIRS NewGRF path
and parsed version, OpenTTD executable path, OpenTTD-LE version, generated
`openttd.cfg`, runtime SHA-256 fingerprints, and the ephemeral `game_port` /
`admin_port` in `summary.json` and `launch.json`. Ports are intentionally not
fixed; they are runtime isolation metadata, not part of the deterministic
scenario definition. `runtime.cfg_effective_sha256` normalizes those ephemeral
ports before hashing the config.

Research runs write:

- `observations.jsonl`
- `actions.jsonl`
- `rewards.jsonl`
- `firs_trace.jsonl`
- `summary.json`
- `report.xlsx`
- `openttd.cfg`
- `replay.json`

`build_cargo_route()` attempts real OpenTTD stations, roads, depots, vehicles,
and orders with `allow_virtual=False`, returning typed failures when the bridge
cannot build a continuous operational route.

## Real Gameplay Video

For visible OpenTTD video, use the live/replay path:

```bash
openttd-le play-firs-live --workbook scenario.xlsx --model gpt-5.5 --record --out runs_firs
openttd-le export-replay --run runs_firs/<timestamp>_firs_ops
openttd-le play-replay --replay runs_firs/<timestamp>_firs_ops/replay.json --workbook scenario.xlsx --out runs_replay
```

Research `eval` runs already write `replay.json`, so they can be rendered later
without rerunning the model:

```bash
openttd-le play-replay --replay runs_openttd/<timestamp>_firs_research/replay.json --workbook scenario.xlsx --out runs_replay
```

Artifacts include `gameplay.mp4` and `gameplay_8x.mp4` when `ffmpeg` is
available and the OpenTTD client window can be captured.

## OpenTTD Benchmarks

Batch model/task comparisons use:

```bash
openttd-le firs-benchmark --workbook scenario.xlsx --tasks lab_supply_mine_short,lab_raw_to_processor --models gpt-5.5 --repeats 3 --openttd-user-dir .openttd
```

Benchmark task definitions live in `scenarios/firs_benchmarks.json`.

Physical construction reliability is measured separately:

```bash
openttd-le benchmark-route-builder --workbook scenario.xlsx --attempts 20 --openttd-user-dir .openttd
```

## Mock Backend

The mock backend is a deterministic Python logistics simulator. It is useful for
unit tests and interface experiments, not for serious OpenTTD research. Its
observations include a `candidate_actions` frontier with executable actions,
feasibility flags, route economics, objective relevance, and ranking metadata.
The environment also exposes:

```python
env.candidate_actions()
env.preview({"type": "build_route", ...})
result = env.step(action)
result.info["reward_details"]
```

Every mock `eval --backend toy` run writes traces:

- `episode.jsonl` with before/after observations, candidates, chosen action,
  preview, reward details, and step info
- `candidate_actions.jsonl`
- `observations.jsonl`
- `rewards.jsonl`
- `diagnostics.jsonl`
- `replay.json`

Export traces for offline training or analysis:

```bash
openttd-le export-dataset --run runs/<run_dir> --out dataset.jsonl
openttd-le export-dataset --run runs --out dataset.jsonl
openttd-le export-dataset --run runs --out dataset.parquet --format parquet
```

Run the mock benchmark suite:

```bash
openttd-le benchmark-core --suite core --agents random,greedy,candidate_rank,preview_rerank --seeds 1,2,3 --out runs_core
```

Use the optional Gymnasium adapter for the mock backend:

```python
from openttd_le.adapters.gymnasium import OpenTTDLEGymEnv

env = OpenTTDLEGymEnv(task_id="coal_easy_001")
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(0)
```

### Mock Procedural Benchmarking

The fixed lab-play suite is useful for debugging but can saturate quickly.
`v0.4` adds deterministic procedural scenario families with explicit
`train` / `dev` / `test` splits:

- `single_route`: generated source-to-sink cargo problems
- `low_cash`: financing-constrained route starts
- `multi_route`: mixed two-route expansion maps
- `chain`: two-stage raw-to-processed logistics topologies

List generated tasks:

```bash
openttd-le list-procedural-scenarios --split dev --count-per-family 2
```

Run the anti-saturation benchmark suite:

```bash
openttd-le benchmark-core --suite procedural --split dev --agents candidate_rank,preview_rerank --seeds 1,2,3 --out runs_procedural
openttd-le benchmark-core --suite procedural --split test --agents my_agent --seeds 1,2,3 --out runs_test
```

Generated scenarios use stable IDs such as `proc_dev_chain_001` and include
split/family/seed tags in observations and artifacts.

### Mock Replay Rendering

Toy/procedural runs write `episode.jsonl` and `replay.json`. Render schematic
audit frames later without rerunning the agent:

```bash
openttd-le render-core-replay --episode runs/<run_dir>/episode.jsonl --out frames/
openttd-le render-core-replay --replay runs/<run_dir>/replay.json --out frames/
openttd-le render-core-replay --replay runs/<run_dir>/replay.json --out replay.mp4 --fps 1
```

SVG frames and `index.html` are always produced. MP4 output here is a schematic
animation, not OpenTTD gameplay footage.

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

`openttd` is the default researcher path for `eval`. It launches real OpenTTD
through the FIRS bridge. `toy` is an explicit mock backend:

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

For Gym/Farama-style research runs, use `eval`; it routes a separate baseline agent
through the real OpenTTD/FIRS environment by default:

```bash
$env:OPENAI_API_KEY="..."
openttd-le eval --scenario open_play_network_value --workbook scenario.xlsx --model gpt-5.5 --max-steps 32 --openttd-user-dir .openttd
openttd-le eval --scenario lab_supply_mine_short --workbook scenario.xlsx --model gpt-5.5 --max-steps 8 --openttd-user-dir .openttd
```

The benchmark writes separate JSONL artifacts for observations, actions, and
rewards so model comparisons do not depend on video recording. Research mode
defaults to physical construction: `build_cargo_route` attempts real OpenTTD
stations, roads, depots, and vehicles with `allow_virtual=False`, returning typed
failures when the bridge cannot build a continuous route. The legacy
`firs-research-run` command still runs the older persistent Python REPL agent for
comparison, but it is no longer the main environment contract.

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
