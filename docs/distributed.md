# Distributed Configuration

This document describes how to configure the API for distributed operation.

## Overview

The API supports distributed operation using Redis or MongoDB as a backend for:

- Session storage
- Memory persistence
- Tool usage logging
- Conversation history

This allows multiple API instances to share state and work together in a horizontally scaled environment.

## Redis Configuration

### Environment Variables

To enable distributed operation with Redis, set the following environment variables:

```bash
# Enable distributed mode
API_ENABLE_DISTRIBUTED=true
API_DISTRIBUTED_BACKEND=redis
API_SESSION_STORE=redis

# Redis connection settings
API_REDIS_HOST=localhost
API_REDIS_PORT=6379
API_REDIS_DB=0
API_REDIS_PASSWORD=your_password
API_REDIS_PREFIX=datamcp:

# Memory backend
API_MEMORY_BACKEND=redis
```

### Docker Compose Example

Here's an example Docker Compose configuration for running the API with Redis:

```yaml
version: '3'

services:
  api1:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_ENABLE_DISTRIBUTED=true
      - API_DISTRIBUTED_BACKEND=redis
      - API_SESSION_STORE=redis
      - API_REDIS_HOST=redis
      - API_REDIS_PORT=6379
      - API_REDIS_DB=0
      - API_REDIS_PASSWORD=your_password
      - API_REDIS_PREFIX=datamcp:
      - API_MEMORY_BACKEND=redis
    depends_on:
      - redis

  api2:
    build: .
    ports:
      - "8001:8000"
    environment:
      - API_ENABLE_DISTRIBUTED=true
      - API_DISTRIBUTED_BACKEND=redis
      - API_SESSION_STORE=redis
      - API_REDIS_HOST=redis
      - API_REDIS_PORT=6379
      - API_REDIS_DB=0
      - API_REDIS_PASSWORD=your_password
      - API_REDIS_PREFIX=datamcp:
      - API_MEMORY_BACKEND=redis
    depends_on:
      - redis

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    command: redis-server --requirepass your_password
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

## MongoDB Configuration

### Environment Variables

To enable distributed operation with MongoDB, set the following environment variables:

```bash
# Enable distributed mode
API_ENABLE_DISTRIBUTED=true
API_DISTRIBUTED_BACKEND=mongodb
API_SESSION_STORE=mongodb

# MongoDB connection settings
API_MONGODB_URI=mongodb://localhost:27017
API_MONGODB_DB=datamcp

# Memory backend
API_MEMORY_BACKEND=mongodb
```

### Docker Compose Example

Here's an example Docker Compose configuration for running the API with MongoDB:

```yaml
version: '3'

services:
  api1:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_ENABLE_DISTRIBUTED=true
      - API_DISTRIBUTED_BACKEND=mongodb
      - API_SESSION_STORE=mongodb
      - API_MONGODB_URI=mongodb://mongodb:27017
      - API_MONGODB_DB=datamcp
      - API_MEMORY_BACKEND=mongodb
    depends_on:
      - mongodb

  api2:
    build: .
    ports:
      - "8001:8000"
    environment:
      - API_ENABLE_DISTRIBUTED=true
      - API_DISTRIBUTED_BACKEND=mongodb
      - API_SESSION_STORE=mongodb
      - API_MONGODB_URI=mongodb://mongodb:27017
      - API_MONGODB_DB=datamcp
      - API_MEMORY_BACKEND=mongodb
    depends_on:
      - mongodb

  mongodb:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongodb-data:/data/db

volumes:
  mongodb-data:
```

## Load Balancing

To distribute traffic across multiple API instances, you can use a load balancer like Nginx, HAProxy, or a cloud load balancer.

### Nginx Example

Here's an example Nginx configuration for load balancing:

```nginx
upstream datamcp_api {
    server api1:8000;
    server api2:8000;
    # Add more servers as needed
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://datamcp_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Session Persistence

When using a load balancer, you should ensure that requests from the same client are routed to the same API instance (session persistence or sticky sessions). This is important for WebSocket connections and streaming responses.

### Nginx Example with Sticky Sessions

```nginx
upstream datamcp_api {
    ip_hash;  # This ensures requests from the same IP go to the same server
    server api1:8000;
    server api2:8000;
    # Add more servers as needed
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://datamcp_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring

When running in a distributed environment, it's important to monitor the health and performance of your API instances and backend services.

You can use tools like Prometheus, Grafana, and ELK stack for monitoring and logging.

## Scaling

To scale the API horizontally, you can add more API instances to your deployment. The shared state in Redis or MongoDB ensures that all instances work together seamlessly.

You can also scale the Redis or MongoDB backends for better performance and reliability.
