FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scenarios ./scenarios
COPY docs ./docs
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["openttd-le"]
CMD ["list-scenarios"]
