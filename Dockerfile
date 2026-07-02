FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scenarios ./scenarios
COPY docs ./docs
COPY openttd_bridge ./openttd_bridge
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["tycoonle-openttd"]
CMD ["list-scenarios"]
