FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Default: run the MCP agent server over streamable-http.
CMD ["python", "-m", "src.mdm.mcp_server", "--http", "--host", "0.0.0.0", "--port", "8000"]
