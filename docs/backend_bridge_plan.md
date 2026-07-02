# OpenTTD Backend Bridge Plan

The benchmark layer is backend-agnostic. The real OpenTTD bridge should satisfy:

```python
state = backend.reset(scenario, seed)
state = backend.apply(action)
```

## Preferred Integration

Use GameScript for in-game goals/actions and the Admin Port for process control
and telemetry.

Current status:

- `tycoonle-openttd smoke-openttd` resolves the installed executable.
- `tycoonle-openttd smoke-openttd --launch` starts a dedicated OpenTTD process in an
  isolated temporary run directory.
- `--backend openttd` does not yet execute gameplay macro-actions.
- `openttd_bridge/OpenTTDLEBridge` contains the initial NoAI bridge skeleton.

Responsibilities:

- load fixed scenario or generated map
- expose towns, industries, cargo acceptance/production
- expose company cash, loan, vehicles, stations, orders, cargo delivered
- apply macro-actions by invoking constrained construction helpers
- advance simulation by a fixed tick/month interval
- emit savegame and screenshots

## Macro-Action Contract

The bridge should implement:

- `build_route(source_id, destination_id, cargo, mode)`
- `add_vehicle(route_id, count)`
- `wait(months)`
- `take_loan(amount)`
- `repay_loan(amount)`

The agent chooses economic intent. The bridge handles legal construction and
returns structured success/failure diagnostics.

## Non-Goals For v0.1

- raw tile-click control
- arbitrary model-generated Squirrel code
- PPO training
- rival companies
- full open-play company growth
