# API Reference

## 📋 Overview

This document provides comprehensive API reference for the DataMCPServerAgent system, including REST endpoints, WebSocket connections, and programmatic interfaces.

## 🌐 REST API Endpoints

### Base URL
```
Production: https://api.datamcp.com/v1
Staging: https://staging-api.datamcp.com/v1
Development: http://localhost:8000/v1
```

### Authentication

#### Bearer Token Authentication
```http
Authorization: Bearer <your-api-token>
```

#### API Key Authentication
```http
X-API-Key: <your-api-key>
```

### 1. Agent Management

#### List Agents
```http
GET /agents
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "research-001",
      "agent_type": "research",
      "status": "active",
      "capabilities": ["web_search", "document_analysis"],
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T14:25:30Z"
    }
  ],
  "total": 5,
  "page": 1,
  "per_page": 10
}
```

#### Get Agent Details
```http
GET /agents/{agent_id}
```

**Response:**
```json
{
  "agent_id": "research-001",
  "agent_type": "research",
  "status": "active",
  "capabilities": ["web_search", "document_analysis"],
  "configuration": {
    "max_concurrent_tasks": 5,
    "timeout": 300
  },
  "metrics": {
    "requests_processed": 1250,
    "average_response_time": 2.3,
    "error_rate": 0.02
  },
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T14:25:30Z"
}
```

#### Create Agent Request
```http
POST /agents/{agent_id}/requests
```

**Request Body:**
```json
{
  "task_type": "research",
  "parameters": {
    "query": "artificial intelligence trends 2024",
    "sources": ["academic", "news", "reports"],
    "max_results": 50
  },
  "priority": 1,
  "timeout": 300,
  "callback_url": "https://your-app.com/webhooks/agent-response"
}
```

**Response:**
```json
{
  "request_id": "req-12345",
  "agent_id": "research-001",
  "status": "queued",
  "estimated_completion": "2024-01-15T14:35:00Z",
  "created_at": "2024-01-15T14:30:00Z"
}
```

#### Get Request Status
```http
GET /agents/{agent_id}/requests/{request_id}
```

**Response:**
```json
{
  "request_id": "req-12345",
  "agent_id": "research-001",
  "status": "completed",
  "result": {
    "summary": "AI trends analysis completed",
    "findings": [
      {
        "trend": "Generative AI adoption",
        "confidence": 0.95,
        "sources": 15
      }
    ],
    "sources_analyzed": 47,
    "processing_time": 45.2
  },
  "created_at": "2024-01-15T14:30:00Z",
  "completed_at": "2024-01-15T14:30:45Z"
}
```

### 2. Pipeline Management

#### List Pipelines
```http
GET /pipelines
```

**Query Parameters:**
- `status`: Filter by status (active, inactive, failed)
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 10, max: 100)

**Response:**
```json
{
  "pipelines": [
    {
      "pipeline_id": "data-processing-001",
      "name": "User Data Processing Pipeline",
      "status": "active",
      "last_run": {
        "run_id": "run-67890",
        "status": "success",
        "start_time": "2024-01-15T12:00:00Z",
        "duration": 1800
      },
      "schedule": "0 */6 * * *",
      "created_at": "2024-01-10T09:00:00Z"
    }
  ],
  "total": 12,
  "page": 1,
  "per_page": 10
}
```

#### Create Pipeline
```http
POST /pipelines
```

**Request Body:**
```json
{
  "name": "Customer Data ETL Pipeline",
  "description": "Extract, transform, and load customer data",
  "schedule": "0 2 * * *",
  "tasks": [
    {
      "task_id": "extract_customers",
      "task_type": "ingestion",
      "parameters": {
        "source_config": {
          "type": "database",
          "connection_string": "postgresql://...",
          "query": "SELECT * FROM customers WHERE updated_at > :last_run"
        }
      }
    },
    {
      "task_id": "transform_data",
      "task_type": "transformation",
      "depends_on": ["extract_customers"],
      "parameters": {
        "transformations": [
          {
            "type": "clean_email",
            "column": "email"
          },
          {
            "type": "standardize_phone",
            "column": "phone"
          }
        ]
      }
    }
  ],
  "notifications": {
    "on_failure": {
      "type": "email",
      "recipients": ["admin@company.com"]
    }
  }
}
```

**Response:**
```json
{
  "pipeline_id": "pipeline-98765",
  "name": "Customer Data ETL Pipeline",
  "status": "inactive",
  "created_at": "2024-01-15T14:30:00Z",
  "tasks": [
    {
      "task_id": "extract_customers",
      "task_type": "ingestion",
      "status": "pending"
    },
    {
      "task_id": "transform_data",
      "task_type": "transformation",
      "status": "pending"
    }
  ]
}
```

#### Trigger Pipeline
```http
POST /pipelines/{pipeline_id}/trigger
```

**Request Body:**
```json
{
  "parameters": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-15"
  },
  "triggered_by": "manual"
}
```

**Response:**
```json
{
  "run_id": "run-54321",
  "pipeline_id": "pipeline-98765",
  "status": "running",
  "triggered_by": "manual",
  "start_time": "2024-01-15T14:35:00Z",
  "estimated_completion": "2024-01-15T15:05:00Z"
}
```

#### Get Pipeline Run Status
```http
GET /pipelines/{pipeline_id}/runs/{run_id}
```

**Response:**
```json
{
  "run_id": "run-54321",
  "pipeline_id": "pipeline-98765",
  "status": "running",
  "progress": 0.6,
  "start_time": "2024-01-15T14:35:00Z",
  "estimated_completion": "2024-01-15T15:05:00Z",
  "tasks": [
    {
      "task_id": "extract_customers",
      "status": "success",
      "start_time": "2024-01-15T14:35:00Z",
      "end_time": "2024-01-15T14:45:00Z",
      "duration": 600,
      "records_processed": 10000
    },
    {
      "task_id": "transform_data",
      "status": "running",
      "start_time": "2024-01-15T14:45:00Z",
      "progress": 0.7,
      "records_processed": 7000
    }
  ]
}
```

### 3. Memory Management

#### Store Memory
```http
POST /memory
```

**Request Body:**
```json
{
  "agent_id": "research-001",
  "memory_type": "episodic",
  "content": {
    "event": "research_completed",
    "query": "AI trends 2024",
    "results_count": 47,
    "confidence": 0.95
  },
  "importance": 0.8,
  "tags": ["research", "ai", "trends"],
  "ttl": 86400
}
```

**Response:**
```json
{
  "memory_id": "mem-abc123",
  "agent_id": "research-001",
  "memory_type": "episodic",
  "stored_at": "2024-01-15T14:30:00Z",
  "expires_at": "2024-01-16T14:30:00Z"
}
```

#### Retrieve Memories
```http
GET /memory
```

**Query Parameters:**
- `agent_id`: Filter by agent ID
- `memory_type`: Filter by memory type
- `tags`: Filter by tags (comma-separated)
- `importance_min`: Minimum importance score
- `limit`: Maximum number of results (default: 10, max: 100)

**Response:**
```json
{
  "memories": [
    {
      "memory_id": "mem-abc123",
      "agent_id": "research-001",
      "memory_type": "episodic",
      "content": {
        "event": "research_completed",
        "query": "AI trends 2024",
        "results_count": 47,
        "confidence": 0.95
      },
      "importance": 0.8,
      "tags": ["research", "ai", "trends"],
      "created_at": "2024-01-15T14:30:00Z"
    }
  ],
  "total": 1,
  "limit": 10
}
```

#### Search Memories
```http
POST /memory/search
```

**Request Body:**
```json
{
  "query": "artificial intelligence research",
  "filters": {
    "agent_id": "research-001",
    "memory_type": ["episodic", "semantic"],
    "importance_min": 0.5,
    "date_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-15T23:59:59Z"
    }
  },
  "limit": 20,
  "include_similarity_score": true
}
```

**Response:**
```json
{
  "memories": [
    {
      "memory_id": "mem-abc123",
      "similarity_score": 0.92,
      "agent_id": "research-001",
      "memory_type": "episodic",
      "content": {
        "event": "research_completed",
        "query": "AI trends 2024"
      },
      "importance": 0.8,
      "created_at": "2024-01-15T14:30:00Z"
    }
  ],
  "total": 1,
  "query_time": 0.045
}
```

### 4. System Monitoring

#### System Health
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:30:00Z",
  "uptime": 86400,
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "healthy",
      "response_time": 0.012,
      "connections": 15
    },
    "cache": {
      "status": "healthy",
      "response_time": 0.003,
      "memory_usage": 0.45
    },
    "agents": {
      "status": "healthy",
      "active_agents": 5,
      "total_agents": 8
    }
  }
}
```

#### System Metrics
```http
GET /metrics
```

**Response:**
```json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "system": {
    "cpu_usage": 0.35,
    "memory_usage": 0.62,
    "disk_usage": 0.28
  },
  "agents": {
    "total_requests": 15420,
    "requests_per_second": 12.5,
    "average_response_time": 2.3,
    "error_rate": 0.02
  },
  "pipelines": {
    "active_pipelines": 3,
    "completed_runs_today": 24,
    "failed_runs_today": 1,
    "success_rate": 0.96
  },
  "memory": {
    "total_memories": 50000,
    "memories_added_today": 1200,
    "cache_hit_rate": 0.85
  }
}
```

## 🔌 WebSocket API

### Connection
```javascript
const ws = new WebSocket('wss://api.datamcp.com/v1/ws');

// Authentication
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your-api-token'
    }));
};
```

### Real-time Agent Updates
```javascript
// Subscribe to agent updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'agent_updates',
    agent_id: 'research-001'
}));

// Receive updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'agent_update') {
        console.log('Agent status:', data.status);
        console.log('Progress:', data.progress);
    }
};
```

### Real-time Pipeline Updates
```javascript
// Subscribe to pipeline updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'pipeline_updates',
    pipeline_id: 'pipeline-98765'
}));

// Receive updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'pipeline_update') {
        console.log('Pipeline status:', data.status);
        console.log('Tasks completed:', data.completed_tasks);
    }
};
```

## 🐍 Python SDK

### Installation
```bash
uv pip install datamcp-sdk
```

### Basic Usage
```python
from datamcp import DataMCPClient

# Initialize client
client = DataMCPClient(
    api_url="https://api.datamcp.com/v1",
    api_token="your-api-token"
)

# Create agent request
request = await client.agents.create_request(
    agent_id="research-001",
    task_type="research",
    parameters={
        "query": "machine learning applications",
        "max_results": 20
    }
)

# Wait for completion
result = await client.agents.wait_for_completion(
    agent_id="research-001",
    request_id=request.request_id,
    timeout=300
)

print(f"Research completed: {result.summary}")
```

### Pipeline Management
```python
# Create pipeline
pipeline = await client.pipelines.create(
    name="Data Processing Pipeline",
    tasks=[
        {
            "task_id": "extract_data",
            "task_type": "ingestion",
            "parameters": {"source": "database"}
        }
    ]
)

# Trigger pipeline
run = await client.pipelines.trigger(pipeline.pipeline_id)

# Monitor progress
async for update in client.pipelines.stream_updates(run.run_id):
    print(f"Progress: {update.progress:.1%}")
    if update.status in ["success", "failed"]:
        break
```

### Memory Operations
```python
# Store memory
memory = await client.memory.store(
    agent_id="research-001",
    memory_type="episodic",
    content={"event": "analysis_completed"},
    importance=0.8
)

# Search memories
memories = await client.memory.search(
    query="data analysis",
    filters={"agent_id": "research-001"},
    limit=10
)

for memory in memories:
    print(f"Memory: {memory.content}")
    print(f"Similarity: {memory.similarity_score:.2f}")
```

## 📊 Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "task_type",
      "reason": "must be one of: research, trading, seo"
    },
    "request_id": "req-error-123",
    "timestamp": "2024-01-15T14:30:00Z"
  }
}
```

### Common Error Codes
- `AUTHENTICATION_ERROR` (401): Invalid or missing authentication
- `AUTHORIZATION_ERROR` (403): Insufficient permissions
- `VALIDATION_ERROR` (400): Invalid request parameters
- `NOT_FOUND` (404): Resource not found
- `RATE_LIMIT_EXCEEDED` (429): Rate limit exceeded
- `INTERNAL_ERROR` (500): Internal server error
- `SERVICE_UNAVAILABLE` (503): Service temporarily unavailable

### Rate Limiting
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642262400
```

## 🔐 Security

### API Token Management
```http
POST /auth/tokens
```

**Request Body:**
```json
{
  "name": "Production API Token",
  "permissions": ["agents:read", "agents:write", "pipelines:read"],
  "expires_at": "2024-12-31T23:59:59Z"
}
```

### Webhook Security
All webhook payloads are signed with HMAC-SHA256:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(f"sha256={expected}", signature)
```

This API reference provides comprehensive documentation for integrating with the DataMCPServerAgent system through REST APIs, WebSocket connections, and programmatic SDKs.
