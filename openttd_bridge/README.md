# OpenTTD Bridge

`OpenTTDLEBridge` is the OpenTTD-side NoAI bridge target. It currently executes
a visible speedrun macro-plan so we can verify that Python can launch OpenTTD
and watch an agent act in the real game: it chooses two towns, labels them with
map signs, and rapidly builds road bursts near the selected towns.

Planned responsibilities:

- publish company/town/industry/vehicle state for the Python backend
- receive validated macro-actions from the Python backend
- build constrained routes using the NoAI construction APIs
- report success/failure diagnostics for each macro-action

Install target, depending on platform:

```text
Documents/OpenTTD/ai/OpenTTDLEBridge/
```

The Python `openttd` backend currently launches OpenTTD and verifies process
integration. The `watch-gpt` command installs this bridge and opens a visible
game:

```bash
tycoonle-openttd watch-gpt
```

The next bridge milestone is to make this AI accept externally supplied
`build_route` commands and return structured success/failure results.
