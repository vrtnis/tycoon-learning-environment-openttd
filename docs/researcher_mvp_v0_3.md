# OpenTTD-LE v0.3 Researcher MVP

The v0.3 MVP keeps the native OpenTTD-LE environment as the source of truth and
adds compatibility layers for common research workflows.

## Native Core

```python
from openttd_le.core.env import OpenTTDLEnv

env = OpenTTDLEnv()
obs, info = env.reset("coal_easy_001", seed=1)
candidates = env.candidate_actions()
preview = env.preview(candidates[0]["action"])
result = env.step(candidates[0]["action"])
```

The native observation, candidate action, preview, reward, episode, replay, and
dataset records carry explicit schema identifiers from
`openttd_le.core.schemas`.

## Benchmark Suite

```bash
openttd-le benchmark-core \
  --suite core \
  --agents random,greedy,candidate_rank,preview_rerank \
  --seeds 1,2,3 \
  --out runs_core
```

The command writes normal per-run artifacts and a `benchmark_summary.json` with
per-agent aggregates: success rate, mean/median/std score, mean delivered cargo,
and mean invalid actions.

## Dataset Export

```bash
openttd-le export-dataset --run runs_core --out dataset.jsonl
openttd-le export-dataset --run runs_core --out dataset.parquet --format parquet
```

JSONL has no optional dependencies. Parquet requires:

```bash
python -m pip install -e .[parquet]
```

## Gymnasium Adapter

```bash
python -m pip install -e .[gymnasium]
```

```python
from openttd_le.adapters.gymnasium import OpenTTDLEGymEnv

env = OpenTTDLEGymEnv(task_id="coal_easy_001", max_candidates=24)
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(0)
```

The Gymnasium adapter uses `Discrete(max_candidates)`. Each integer chooses an
index into the current native `candidate_actions` frontier, which is returned in
`info["candidate_actions"]`. This keeps standard RL loops simple while
preserving the richer native contract.
