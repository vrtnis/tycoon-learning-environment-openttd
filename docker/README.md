# Docker

Build the TycoonLE OpenTTD image:

```bash
docker build -t tycoonle-openttd:0.1 .
```

Run a baseline:

```bash
docker run --rm -v "%cd%/runs:/app/runs" tycoonle-openttd:0.1 \
  eval --agent greedy --scenario coal_easy_001 --out runs
```

This image currently runs the deterministic `toy` backend. A later image will
add pinned OpenTTD binaries/assets and the GameScript/Admin Port bridge.
