# TycoonLE OpenTTD Determinism Contract

This contract defines the claim made by `OpenTTD-FIRS-Deterministic-v0`.

For a fixed environment version, scenario/workbook, OpenTTD executable, NewGRF/base set content, seed, and action sequence, repeated runs must produce the same public Gymnasium trace.

## Public Trace

The strict determinism check compares:

- encoded observations
- action masks
- candidate ordering
- selected action indices
- selected action payloads
- rewards
- `terminated` and `truncated`
- route outcomes exposed through public `info`
- cargo delivered
- money and profit values exposed through public `info`

Only runtime artifacts may differ:

- run directories
- process ids
- ephemeral ports
- timestamps
- absolute local paths

If a simulator field cannot be made deterministic, it should not be exposed by the deterministic Gym adapter. The relaxed `semantic` trace mode is retained only for development smoke checks and is not a Farama-grade determinism claim.

## Runtime Lock

Each deterministic run writes `runtime_lock.json` from the FIRS environment and the determinism harness writes `determinism_runtime_lock.json` next to the trace artifacts.

The comparable runtime lock includes:

- TycoonLE OpenTTD package version
- Python version
- OpenTTD executable SHA256
- OpenGFX/base set SHA256
- FIRS NewGRF version and SHA256
- workbook SHA256
- benchmark/scenario SHA256
- normalized `openttd.cfg` SHA256
- installed and source GameScript/AI bridge SHA256
- normalized OpenTTD command line
- seed, map settings, economy, target chain, task id, and Gym env id

Exact paths, ports, and process-local command details remain in the audit record but are excluded from comparable lock equality.

## Harness

Use:

```powershell
tycoonle-openttd determinism-check `
  --workbook scenario.xlsx `
  --scenario lab_raw_to_processor_short `
  --executable tools\openttd-15.3\openttd-15.3-windows-win64\openttd.exe `
  --openttd-user-dir .openttd `
  --agent first_valid `
  --seed 1 `
  --repeats 5 `
  --max-steps 3 `
  --trace-mode strict
```

By default the first repeat generates the action-index script and later repeats replay that same script. This matches the Farama-style requirement: same seed plus same actions must produce the same observation, reward, termination, truncation, and info trace.

The command exits non-zero on the first trace or runtime-lock mismatch and writes:

- `determinism_report.json`
- per-run `determinism_raw_trace.json`
- per-run `determinism_normalized_trace.json`
- per-run `determinism_runtime_lock.json`
