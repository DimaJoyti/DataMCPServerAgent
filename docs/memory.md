# Memory Systems

This document provides detailed information about the memory systems in the DataMCPServerAgent project.

## Overview

The DataMCPServerAgent project implements several memory systems with increasing levels of sophistication:

1. **In-Memory Storage**: Simple in-memory storage for the basic agent
2. **Persistent Storage**: Database-backed storage for the enhanced agent
3. **Context-Aware Memory**: Semantic search-enabled memory for the advanced enhanced agent
4. **Collaborative Knowledge Base**: Shared knowledge storage for the multi-agent learning system

## In-Memory Storage

The basic agent uses in-memory storage for conversation history, tool usage, and entities. This memory is implemented in `src/agents/agent_architecture.py` as the `AgentMemory` class.

Key features:
- Stores conversation history as a list of messages
- Tracks tool usage history
- Maintains entity memory
- Limited by session duration (memory is lost when the agent is restarted)

Example usage:
```python
from src.agents.agent_architecture import AgentMemory

# Create memory instance
memory = AgentMemory(max_history_length=50)

# Add a message to memory
memory.add_message({"role": "user", "content": "Hello, agent!"})

# Get recent messages
recent_messages = memory.get_recent_messages(3)
```

## Persistent Storage

The enhanced agent uses persistent storage for memory, implemented in `src/memory/memory_persistence.py` as the `MemoryDatabase` class.

Key features:
- Persists memory between sessions
- Supports multiple storage backends (SQLite, file-based)
- Stores conversation history, tool usage, and entities
- Provides methods for querying and updating memory

Example usage:
```python
from src.memory.memory_persistence import MemoryDatabase

# Create memory database
db = MemoryDatabase("agent_memory.db")

# Store a conversation message
db.store_conversation_message("user", "Hello, agent!")

# Get recent conversation history
history = db.get_conversation_history(limit=5)
```

### Storage Backends

The persistent memory system supports multiple storage backends:

#### SQLite Backend

The SQLite backend stores memory in a SQLite database file. This is the default backend.

Configuration:
```python
db = MemoryDatabase("agent_memory.db", backend="sqlite")
```

#### File-Based Backend

The file-based backend stores memory as JSON files in a directory structure.

Configuration:
```python
db = MemoryDatabase("agent_memory", backend="file")
```

#### Redis Backend

The Redis backend stores memory in a Redis database, enabling distributed memory access.

Configuration:
```python
db = MemoryDatabase(
    "agent_memory",
    backend="redis",
    redis_config={
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": ""
    }
)
```

#### MongoDB Backend

The MongoDB backend stores memory in a MongoDB database, enabling distributed memory access with advanced querying capabilities.

Configuration:
```python
db = MemoryDatabase(
    "agent_memory",
    backend="mongodb",
    mongodb_config={
        "uri": "mongodb://localhost:27017",
        "db": "agent_memory"
    }
)
```

## Context-Aware Memory

The advanced enhanced agent uses context-aware memory with semantic search, implemented in `src/memory/context_aware_memory.py` as the `ContextManager` and `MemoryRetriever` classes.

Key features:
- Retrieves relevant information based on the current request
- Uses semantic search to find related conversation history
- Maintains working memory for the current session
- Extracts entities from user requests

Example usage:
```python
from src.memory.context_aware_memory import ContextManager, MemoryRetriever
from src.memory.memory_persistence import MemoryDatabase

# Create memory database
db = MemoryDatabase("agent_memory.db")

# Create memory retriever
retriever = MemoryRetriever(db)

# Create context manager
context_manager = ContextManager(retriever)

# Get relevant context for a request
context = await context_manager.get_context("Tell me about the iPhone 14")
```

## Collaborative Knowledge Base

The multi-agent learning system uses a collaborative knowledge base, implemented in `src/memory/collaborative_knowledge.py` as the `CollaborativeKnowledgeBase` class.

Key features:
- Stores knowledge shared between agents
- Tracks knowledge transfers between agents
- Assigns knowledge to agents with proficiency levels
- Provides methods for querying and updating knowledge

Example usage:
```python
from src.memory.collaborative_knowledge import CollaborativeKnowledgeBase
from src.memory.memory_persistence import MemoryDatabase

# Create memory database
db = MemoryDatabase("agent_memory.db")

# Create collaborative knowledge base
knowledge_base = CollaborativeKnowledgeBase(db)

# Store knowledge
knowledge_id = knowledge_base.store_knowledge(
    {
        "content": "The iPhone 14 was released in September 2022",
        "confidence": 0.9,
        "domain": "product_information",
        "applicability": ["search_agent", "product_agent"],
        "prerequisites": []
    },
    source_agent="search_agent"
)

# Assign knowledge to an agent
knowledge_base.assign_knowledge_to_agent("product_agent", knowledge_id, 0.7)

# Get knowledge for an agent
agent_knowledge = knowledge_base.get_agent_knowledge("product_agent")
```

## Memory Configuration

Memory systems can be configured using environment variables in the `.env` file:

```
# Memory Configuration
MEMORY_DB_PATH=agent_memory.db
MEMORY_TYPE=sqlite  # Options: sqlite, file, redis, mongodb

# Redis Configuration (if using Redis for memory)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# MongoDB Configuration (if using MongoDB for memory)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=agent_memory
```

These environment variables are accessed through the `src/utils/env_config.py` module:

```python
from src.utils.env_config import get_memory_config

# Get memory configuration
memory_config = get_memory_config()

# Create memory database with configuration
db = MemoryDatabase(
    memory_config["db_path"],
    backend=memory_config["memory_type"],
    redis_config=memory_config.get("redis"),
    mongodb_config=memory_config.get("mongodb")
)
```