# OpenTTD-LE v0.4 Procedural Benchmarking

The v0.4 layer reduces benchmark saturation risk by adding generated scenario
families with deterministic split seeds. The native environment contract is
unchanged: generated tasks are ordinary `Scenario` objects, so existing agents,
artifacts, dataset export, and Gymnasium adapters continue to work.

## Splits

Procedural tasks are generated from disjoint seed bands:

- `train`: for agent development and learning
- `dev`: for tuning and local comparison
- `test`: for held-out reporting

Do not tune prompts, value models, or heuristics directly on `test`.

## Families

- `single_route`: one generated producer-to-consumer route with distractors
- `low_cash`: the route is feasible only with financing discipline
- `multi_route`: mixed opportunities require building a useful two-route network
- `chain`: raw-to-processor and processor-to-sink topology

Generated IDs are stable for a split and count:

```text
proc_dev_single_route_001
proc_dev_low_cash_001
proc_dev_multi_route_001
proc_dev_chain_001
```

## Commands

List generated scenarios:

```bash
openttd-le list-procedural-scenarios --split dev --count-per-family 3
```

Run the procedural benchmark:

```bash
openttd-le benchmark-core \
  --suite procedural \
  --split dev \
  --agents random,greedy,candidate_rank,preview_rerank \
  --seeds 1,2,3 \
  --procedural-count-per-family 3 \
  --out runs_procedural_dev
```

Use the held-out split for final reports:

```bash
openttd-le benchmark-core \
  --suite procedural \
  --split test \
  --agents candidate_rank,preview_rerank \
  --seeds 1,2,3 \
  --procedural-count-per-family 3 \
  --out runs_procedural_test
```

## Why This Matters

Fixed maps are easy to overfit. Procedural families force agents to learn or
reason over logistics structure: cost, distance, budget, route count, cargo
value, and dependency topology. The generated suite is still abstract and fast;
physical OpenTTD/FIRS remains an experimental backend until construction
reliability is benchmarked separately.
