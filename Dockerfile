# PJ-03 REST API (research prototype).
# Build:  docker build -t pj03-api .
# Run:    docker run -p 8000:8000 pj03-api
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps .

# Model artifact is trained at first start (CPU, ~30s) if not mounted.
VOLUME ["/app/artifacts", "/app/data"]

EXPOSE 8000
CMD ["python", "scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]
