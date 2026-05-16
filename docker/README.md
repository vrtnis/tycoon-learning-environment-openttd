# Docker

Build the benchmark scaffold:

```bash
docker build -t openttd-le:0.1 .
```

Run a baseline:

```bash
docker run --rm -v "%cd%/runs:/app/runs" openttd-le:0.1 \
  eval --agent greedy --scenario coal_easy_001 --out runs
```

This image currently runs the deterministic `toy` backend. A later image will
add pinned OpenTTD binaries/assets and the GameScript/Admin Port bridge.
