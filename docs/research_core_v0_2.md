# TycoonLE OpenTTD v0.2 Research Core

The v0.2 research core separates the environment substrate from agent research.
It keeps the simulator contract small while exposing enough structure for
planning, search, replay, and offline learning.

## Contract

```python
obs, info = env.reset("coal_easy_001", seed=1)
candidates = env.candidate_actions()
preview = env.preview(candidates[0]["action"])
result = env.step(candidates[0]["action"])
reward_details = result.info["reward_details"]
```

Observations include `candidate_actions`. Each candidate has:

- `id`: stable action identifier
- `kind`: action family such as `build_route`, `add_vehicle`, or `wait`
- `action`: executable action payload
- `feasible`: whether the action is feasible under available financing
- `directly_executable`: whether it can be executed without a prior finance action
- `requires_loan`: cash shortfall for direct execution
- `rank_score`: baseline ranking signal
- `estimates`: local route or action estimates
- `diagnostics`: structured warnings or failure reasons

## Artifacts

Every `tycoonle-openttd eval` run emits:

- `episode.jsonl`: joined before/action/after/reward rows
- `candidate_actions.jsonl`: the action frontier available at each step
- `observations.jsonl`: before/after observations without embedded candidates
- `rewards.jsonl`: decomposed reward components and milestones
- `diagnostics.jsonl`: preview and reward diagnostics
- `replay.json`: compact action/reward replay manifest

Dataset export is local and file-based:

```bash
tycoonle-openttd export-dataset --run runs --out dataset.jsonl
```

The exported JSONL rows are intended for value-model training, action reranking,
imitation learning, and counterfactual analysis.

## Boundary

Reference agents may consume `candidate_actions`, but the environment does not
solve planning internally. The core substrate provides state, legal options,
preview estimates, rewards, diagnostics, and replay data. Researchers bring the
policy, search procedure, learned value function, or relabeling algorithm.
