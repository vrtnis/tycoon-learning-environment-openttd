# OpenTTD-LE v0.1 Draft Report

OpenTTD-LE evaluates planning and LLM agents on transport-logistics tasks:
route selection, fleet expansion, debt control, and long-horizon network growth.

## Quickstart

```bash
python -m pip install -e .
openttd-le eval --agent greedy --scenario coal_easy_001
```

## Environment

The v0.1 scaffold exposes a Gym-style loop:

```python
obs, info = env.reset("coal_easy_001", seed=1)
result = env.step({"type": "wait", "months": 3})
```

Observations include:

- scenario task and goals
- map summary
- towns and industries
- company cash and loan
- routes and vehicle counts
- score breakdown and recent event

Macro-actions include:

- `build_route`
- `add_vehicle`
- `wait`
- `take_loan`
- `repay_loan`

## Scenarios

The bundled lab-play pack currently contains 10 fixed tasks spanning coal, oil,
wood, grain, goods, passengers, mail, low-cash starts, terrain cost, and
multi-route expansion.

## Agents

Included agents:

- random macro-agent
- greedy route-return baseline
- OpenAI Responses API agent
- OpenRouter chat-completions agent

## Artifacts

Each run emits:

- `summary.json`
- `actions.jsonl`
- `metrics.csv`
- `final_state.json`
- `screenshots/final_map.svg`
- `agent_trace.md`

## Current Limitation

This release uses a deterministic logistics backend to validate the benchmark
contract. The real OpenTTD backend remains the next milestone. The intended
bridge boundary is already represented by `Backend.reset()` and `Backend.apply()`.

## Differentiation

Unlike prior tycoon RL work focused on amusement-park placement, OpenTTD-LE is
organized around transport-network logistics. Unlike FLE, which emphasizes
factory automation, OpenTTD-LE emphasizes route economics, fleet operation,
capital constraints, and network expansion.
