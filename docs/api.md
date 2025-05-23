# API Documentation

This document describes the API for DataMCPServerAgent.

## Overview

The DataMCPServerAgent API provides a RESTful interface to interact with the agent system. It allows you to:

- Chat with agents
- Manage agent instances
- Access and manipulate memory
- Execute tools
- Check the health of the API

## Getting Started

### Running the API Server

You can run the API server using the `main.py` script with the `--mode=api` option:

```bash
python main.py --mode=api --host 0.0.0.0 --port 8000 --reload --debug
```

Or you can use the dedicated `run_api.py` script:

```bash
python run_api.py --host 0.0.0.0 --port 8000 --reload --debug
```

### API Documentation

Once the API server is running, you can access the API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

The API supports API key authentication. To enable authentication, set the following environment variables:

```bash
API_ENABLE_AUTH=true
API_KEY_HEADER=X-API-Key
API_KEYS=key1,key2,key3
```

When authentication is enabled, you need to include the API key in the request header:

```
X-API-Key: key1
```

## Endpoints

### Root

- `GET /`: Get API information

### Health

- `GET /health`: Check the health of the API

### Agents

- `GET /agents`: List all available agent modes
- `GET /agents/{agent_mode}`: Get information about a specific agent mode
- `POST /agents`: Create a new agent instance

### Chat

- `POST /chat`: Send a message to the agent and get a response
- `POST /chat/stream`: Send a message to the agent and get a streaming response
- `GET /chat/sessions/{session_id}`: Get chat history for a session

### Memory

- `POST /memory`: Store a memory item
- `GET /memory/{session_id}`: Retrieve memory items for a session
- `DELETE /memory/{session_id}`: Clear memory for a session

### Tools

- `GET /tools`: List all available tools
- `POST /tools/execute`: Execute a tool

## Request and Response Models

### Chat

#### ChatRequest

```json
{
  "message": "Hello, agent!",
  "session_id": "optional-session-id",
  "agent_mode": "basic",
  "user_id": "optional-user-id",
  "context": {
    "key": "value"
  },
  "stream": false
}
```

#### ChatResponse

```json
{
  "message_id": "unique-message-id",
  "response": "Hello! How can I help you today?",
  "session_id": "session-id",
  "created_at": "2023-01-01T00:00:00Z",
  "agent_mode": "basic",
  "tool_usage": [
    {
      "tool_name": "web_search",
      "tool_input": {
        "query": "example query"
      },
      "tool_output": "example output"
    }
  ],
  "sources": [
    {
      "url": "https://example.com",
      "title": "Example Page",
      "snippet": "Example snippet"
    }
  ],
  "metadata": {
    "user_id": "user-id",
    "context": {
      "key": "value"
    }
  }
}
```

### Agents

#### AgentRequest

```json
{
  "agent_mode": "basic",
  "config": {
    "key": "value"
  }
}
```

#### AgentResponse

```json
{
  "agent_id": "unique-agent-id",
  "agent_mode": "basic",
  "status": "available",
  "capabilities": ["chat", "web_search", "web_browsing"],
  "created_at": "2023-01-01T00:00:00Z",
  "metadata": {
    "description": "Basic Agent"
  }
}
```

### Memory

#### MemoryRequest

```json
{
  "session_id": "session-id",
  "query": "optional-query",
  "memory_item": {
    "key": "value"
  },
  "memory_backend": "sqlite"
}
```

#### MemoryResponse

```json
{
  "session_id": "session-id",
  "memory_items": [
    {
      "id": "memory-id",
      "content": "Memory content",
      "timestamp": "2023-01-01T00:00:00Z",
      "metadata": {
        "key": "value"
      }
    }
  ],
  "memory_backend": "sqlite",
  "metadata": {
    "operation": "retrieve",
    "query": "optional-query",
    "limit": 10,
    "offset": 0,
    "timestamp": "2023-01-01T00:00:00Z"
  }
}
```

### Tools

#### ToolRequest

```json
{
  "tool_name": "web_search",
  "tool_input": {
    "query": "example query"
  },
  "session_id": "optional-session-id",
  "agent_mode": "basic"
}
```

#### ToolResponse

```json
{
  "tool_name": "web_search",
  "tool_output": "example output",
  "execution_time": 0.5,
  "status": "success",
  "metadata": {
    "session_id": "session-id",
    "agent_mode": "basic",
    "tool_input": {
      "query": "example query"
    }
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error responses:

```json
{
  "error_code": "ERROR_CODE",
  "error_message": "Error message",
  "error_details": {
    "key": "value"
  },
  "request_id": "request-id"
}
```

## Rate Limiting

The API supports rate limiting. To enable rate limiting, set the following environment variables:

```bash
API_ENABLE_RATE_LIMITING=true
API_RATE_LIMIT_PER_MINUTE=60
```

When rate limiting is enabled, the API will return a 429 Too Many Requests response if the client exceeds the rate limit.

## Environment Variables

The API can be configured using environment variables:

### API Settings

- `API_TITLE`: API title
- `API_DESCRIPTION`: API description
- `API_VERSION`: API version
- `API_OPENAPI_URL`: OpenAPI URL
- `API_DOCS_URL`: Swagger UI URL
- `API_REDOC_URL`: ReDoc URL

### Server Settings

- `API_HOST`: Host to bind the server to
- `API_PORT`: Port to bind the server to
- `API_DEBUG`: Enable debug mode
- `API_RELOAD`: Enable auto-reload on code changes

### Security Settings

- `API_ENABLE_AUTH`: Enable authentication
- `API_KEY_HEADER`: API key header
- `API_KEYS`: Comma-separated list of API keys

### CORS Settings

- `API_ALLOW_ORIGINS`: Comma-separated list of allowed origins for CORS
- `API_ALLOW_METHODS`: Comma-separated list of allowed methods for CORS
- `API_ALLOW_HEADERS`: Comma-separated list of allowed headers for CORS

### Rate Limiting

- `API_ENABLE_RATE_LIMITING`: Enable rate limiting
- `API_RATE_LIMIT_PER_MINUTE`: Rate limit per minute

### Logging

- `API_LOG_LEVEL`: Log level

### Agent Settings

- `API_DEFAULT_AGENT_MODE`: Default agent mode
- `API_ENABLE_ALL_TOOLS`: Enable all tools

### Memory Settings

- `API_MEMORY_BACKEND`: Memory backend (sqlite, file, redis, mongodb)

### Redis Settings

- `API_REDIS_HOST`: Redis host
- `API_REDIS_PORT`: Redis port
- `API_REDIS_DB`: Redis database
- `API_REDIS_PASSWORD`: Redis password
- `API_REDIS_PREFIX`: Redis key prefix

### MongoDB Settings

- `API_MONGODB_URI`: MongoDB URI
- `API_MONGODB_DB`: MongoDB database

### Distributed Settings

- `API_ENABLE_DISTRIBUTED`: Enable distributed mode
- `API_DISTRIBUTED_BACKEND`: Distributed backend (redis, mongodb)
- `API_SESSION_STORE`: Session store (redis, mongodb, memory)
