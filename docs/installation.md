# Installation Guide

This document provides instructions for installing the DataMCPServerAgent.

## Prerequisites

- Python 3.8 or higher
- Node.js (for Bright Data MCP)
- Bright Data MCP credentials

## Installing from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/DimaJoyti/DataMCPServerAgent.git
   cd DataMCPServerAgent
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

   Alternatively, you can use the installation script:
   ```bash
   python install_dependencies.py
   ```

## Environment Configuration

Create a `.env` file in the project root by copying the template:

```bash
cp .env.template .env
```

Then edit the `.env` file with your actual credentials:

```
# Bright Data MCP Credentials
API_TOKEN=your_bright_data_api_token
BROWSER_AUTH=your_bright_data_browser_auth
WEB_UNLOCKER_ZONE=your_bright_data_web_unlocker_zone

# Model Configuration
MODEL_NAME=claude-3-5-sonnet-20240620
MODEL_PROVIDER=anthropic

# Memory Configuration
MEMORY_DB_PATH=agent_memory.db
MEMORY_TYPE=sqlite  # Options: sqlite, file, redis, mongodb
```

### Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_TOKEN` | Bright Data API token | Yes |
| `BROWSER_AUTH` | Bright Data browser authentication | Yes |
| `WEB_UNLOCKER_ZONE` | Bright Data web unlocker zone | Yes |
| `MODEL_NAME` | Language model name | No (default: claude-3-5-sonnet-20240620) |
| `MODEL_PROVIDER` | Language model provider | No (default: anthropic) |
| `MEMORY_DB_PATH` | Path to memory database | No (default: agent_memory.db) |
| `MEMORY_TYPE` | Memory storage type | No (default: sqlite) |

## Docker Installation

A Docker image is available for easy deployment:

```bash
docker pull dimajoyti/datamcpserveragent:latest
```

Run the Docker container:

```bash
docker run -p 8000:8000 --env-file .env dimajoyti/datamcpserveragent:latest
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure you have the correct Python version:
   ```bash
   python --version
   ```

2. Check that Node.js is installed:
   ```bash
   node --version
   ```

3. Verify your Bright Data MCP credentials in the `.env` file.

4. If you're using Redis or MongoDB for memory storage, ensure the services are running and accessible.