# Distributed Memory

This document provides detailed information about the distributed memory capabilities in the DataMCPServerAgent project.

## Overview

The distributed memory system extends the agent architecture with scalable memory capabilities. It enables agents to store and retrieve memory across multiple storage backends, providing high availability, fault tolerance, and horizontal scalability.

## Components

The distributed memory system consists of the following components:

### DistributedMemoryManager

The `DistributedMemoryManager` class provides a unified interface for all memory operations, regardless of the backend used. It handles caching, metrics, and error handling.

Key features:
- Unified API for memory operations
- Local caching for performance optimization
- Metrics collection for monitoring
- Error handling and logging
- Support for multiple storage backends

Implementation: `src/memory/distributed_memory_manager.py`

### DistributedMemoryBackend

The `DistributedMemoryBackend` abstract base class defines the interface for all memory backends. It provides methods for saving, loading, and deleting entities, as well as managing conversation history and tool usage.

Key features:
- Abstract interface for memory backends
- Methods for entity management
- Methods for conversation history
- Methods for tool usage tracking

Implementation: `src/memory/distributed_memory.py`

### RedisMemoryBackend

The `RedisMemoryBackend` class implements the distributed memory backend using Redis. It provides high-performance, in-memory storage with optional persistence.

Key features:
- High-performance in-memory storage
- Optional persistence to disk
- Support for complex data structures
- Atomic operations
- Pub/sub capabilities

Implementation: `src/memory/distributed_memory.py`

### MongoDBMemoryBackend

The `MongoDBMemoryBackend` class implements the distributed memory backend using MongoDB. It provides document-oriented storage with rich querying capabilities.

Key features:
- Document-oriented storage
- Rich querying capabilities
- Horizontal scaling through sharding
- Automatic failover with replica sets
- Support for complex data structures

Implementation: `src/memory/distributed_memory.py`

### DistributedMemoryFactory

The `DistributedMemoryFactory` class provides a factory method for creating memory backends based on configuration.

Key features:
- Factory method for creating memory backends
- Support for multiple backend types
- Configuration-based initialization

Implementation: `src/memory/distributed_memory.py`

## Memory Operations

The distributed memory system supports the following operations:

### Entity Management

- **save_entity**: Save an entity to distributed memory
- **load_entity**: Load an entity from distributed memory
- **delete_entity**: Delete an entity from distributed memory

### Conversation History

- **save_conversation_history**: Save conversation history to distributed memory
- **load_conversation_history**: Load conversation history from distributed memory

### Tool Usage

- **save_tool_usage**: Save tool usage to distributed memory

### Memory Summary

- **get_memory_summary**: Generate a summary of the memory contents

### Caching

- **clear_cache**: Clear the local cache

### Metrics

- **get_metrics**: Get memory operation metrics

### Health Check

- **ping**: Check if the memory backend is accessible

## Configuration

The distributed memory system can be configured using the following environment variables:

### Redis Configuration

- `MEMORY_TYPE`: Set to `redis` to use Redis backend
- `REDIS_HOST`: Redis host (default: `localhost`)
- `REDIS_PORT`: Redis port (default: `6379`)
- `REDIS_DB`: Redis database number (default: `0`)
- `REDIS_PASSWORD`: Redis password (default: `None`)
- `REDIS_PREFIX`: Prefix for Redis keys (default: `datamcp`)

### MongoDB Configuration

- `MEMORY_TYPE`: Set to `mongodb` to use MongoDB backend
- `MONGODB_URI`: MongoDB connection string (default: `mongodb://localhost:27017/`)
- `MONGODB_DB`: MongoDB database name (default: `agent_memory`)

### General Configuration

- `MEMORY_NAMESPACE`: Namespace for memory keys to avoid collisions (default: `agent`)

## Scaling Strategies

The distributed memory system supports the following scaling strategies:

### Horizontal Scaling

- **Redis Cluster**: Scale Redis horizontally using Redis Cluster
- **MongoDB Sharding**: Scale MongoDB horizontally using sharding
- **Multiple Instances**: Run multiple instances of the agent with shared memory

### Vertical Scaling

- **Increased Resources**: Allocate more CPU and memory to the memory backend
- **Optimized Configuration**: Tune the memory backend for performance

### Caching Strategies

- **Local Caching**: Cache frequently accessed data locally
- **Time-to-Live (TTL)**: Set expiration times for cached data
- **Cache Invalidation**: Invalidate cache when data changes

## Usage

To use the distributed memory agent, you can run the following command:

```bash
python -m src.core.distributed_memory_main
```

Or use the Python API:

```python
import asyncio
from src.core.distributed_memory_main import chat_with_distributed_memory_agent

asyncio.run(chat_with_distributed_memory_agent())
```

## Example

See `examples/enhanced_distributed_memory_example.py` for a complete example of using the distributed memory system:

```python
import asyncio
from src.memory.distributed_memory_manager import DistributedMemoryManager

# Create distributed memory manager
memory_manager = DistributedMemoryManager(
    memory_type="redis",
    config={
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "prefix": "datamcp:"
    },
    namespace="example"
)

# Save entity
await memory_manager.save_entity(
    "user",
    "user123",
    {
        "name": "John Doe",
        "email": "john@example.com",
        "preferences": {
            "theme": "dark",
            "language": "en"
        }
    }
)

# Load entity
user = await memory_manager.load_entity("user", "user123")

# Delete entity
await memory_manager.delete_entity("user", "user123")

# Get memory summary
summary = await memory_manager.get_memory_summary()
```

## Comparison of Backends

### Redis

Redis is an in-memory data structure store that can be used as a database, cache, and message broker.

Advantages:
- Very high performance (in-memory)
- Support for complex data structures
- Built-in persistence options
- Pub/sub capabilities
- Clustering for horizontal scaling

Disadvantages:
- Limited query capabilities
- Memory-constrained (data size limited by available RAM)
- Less flexible schema compared to MongoDB

### MongoDB

MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like documents.

Advantages:
- Rich query language
- Flexible schema
- Horizontal scaling through sharding
- Automatic failover with replica sets
- Support for large datasets

Disadvantages:
- Lower performance compared to Redis
- More complex setup and maintenance
- Higher resource requirements

## Future Improvements

Potential future improvements to the distributed memory system include:

1. **Multi-Backend Support**: Support for using multiple backends simultaneously
2. **Automatic Failover**: Automatic failover between backends
3. **Data Replication**: Replication of data across multiple backends
4. **Distributed Caching**: Distributed caching using technologies like Memcached
5. **Memory Compression**: Compression of memory data to reduce storage requirements
6. **Memory Encryption**: Encryption of sensitive memory data
7. **Memory Analytics**: Advanced analytics and visualization of memory usage
8. **Memory Pruning**: Automatic pruning of old or unused memory data
9. **Memory Versioning**: Versioning of memory data for rollback capabilities
10. **Memory Synchronization**: Synchronization of memory data across multiple agents
