# Knowledge Graph Integration

This document describes the knowledge graph integration for better context understanding in the DataMCPServerAgent project.

## Overview

The knowledge graph integration enhances the agent's context understanding by representing entities and their relationships in a graph structure. This allows the agent to:

1. Extract entities and relationships from text
2. Store them in a structured graph
3. Query the graph for relevant context
4. Use the context to improve responses

The knowledge graph integration builds on top of the distributed memory system, adding a layer of semantic understanding to the agent's memory.

## Components

### Knowledge Graph

The core component is the `KnowledgeGraph` class, which provides:

- A graph-based representation of entities and relationships
- Storage of nodes (entities) and edges (relationships)
- Persistence using a database backend
- Query capabilities for retrieving relevant context

The knowledge graph uses both NetworkX for in-memory operations and RDFLib for semantic queries using SPARQL.

### Knowledge Graph Manager

The `KnowledgeGraphManager` class provides:

- Entity extraction from text using a language model
- Relationship identification between entities
- Mapping of entities and relationships to the knowledge graph
- Processing of conversation messages and tool usage

### Knowledge Graph Integration

The `KnowledgeGraphIntegration` class integrates the knowledge graph with the distributed memory manager by:

- Intercepting memory operations
- Processing entities, conversation messages, and tool usage
- Providing context for user requests
- Offering query capabilities for the knowledge graph

## Entity Types

The knowledge graph supports the following entity types:

- Person: Individuals mentioned in conversations
- Organization: Companies, groups, or institutions
- Location: Physical places or geographical entities
- Product: Products, services, or offerings
- Event: Occurrences or happenings
- Concept: Abstract ideas or notions
- Document: Files, papers, or written materials
- Website: Web resources or online platforms
- Tool: Tools or utilities used by the agent
- Query: User queries or requests
- Response: Agent responses or answers

## Relationship Types

The knowledge graph supports the following relationship types:

- hasProperty: Entity has a property
- relatedTo: General relationship between entities
- partOf: Entity is part of another entity
- hasPart: Entity contains another entity
- createdBy: Entity was created by another entity
- created: Entity created another entity
- locatedIn: Entity is located in another entity
- contains: Entity contains another entity
- mentionedIn: Entity is mentioned in another entity
- mentions: Entity mentions another entity
- usedBy: Entity is used by another entity
- uses: Entity uses another entity
- precededBy: Entity is preceded by another entity
- follows: Entity follows another entity
- similarTo: Entity is similar to another entity
- sameAs: Entity is the same as another entity
- instanceOf: Entity is an instance of another entity
- hasInstance: Entity has an instance of another entity

## Usage

### Initialization

```python
from src.memory.distributed_memory_manager import DistributedMemoryManager
from src.memory.knowledge_graph_integration import KnowledgeGraphIntegration
from src.memory.memory_persistence import MemoryDatabase
from langchain_anthropic import ChatAnthropic

# Initialize memory database
db = MemoryDatabase("memory.db")

# Initialize distributed memory manager
memory_manager = DistributedMemoryManager(
    memory_type="sqlite",
    config={},
    namespace="example"
)

# Initialize model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Initialize knowledge graph integration
kg_integration = KnowledgeGraphIntegration(
    memory_manager=memory_manager,
    db=db,
    model=model,
    namespace="example"
)
```

### Saving Entities

Entities are automatically processed when saved to the distributed memory:

```python
await memory_manager.save_entity(
    "person",
    "john_doe",
    {
        "name": "John Doe",
        "age": 35,
        "occupation": "Software Engineer"
    }
)
```

### Processing Conversation Messages

Conversation messages are automatically processed when saved:

```python
await memory_manager.save_conversation_message(
    {
        "role": "user",
        "content": "I'm interested in learning about distributed memory systems."
    },
    "conversation_123"
)
```

### Getting Context for a Request

```python
context = await kg_integration.get_context_for_request(
    "How does Redis compare to MongoDB for distributed memory?"
)

# Use context to enhance response
```

### Executing SPARQL Queries

```python
results = await kg_integration.execute_sparql_query("""
    SELECT ?entity ?property ?value
    WHERE {
        ?entity ?property ?value .
        FILTER(CONTAINS(STR(?value), "Redis"))
    }
    LIMIT 10
""")
```

### Getting Entity Context

```python
context = await kg_integration.get_entity_context(
    "person",
    "john_doe",
    max_depth=2
)
```

## Benefits

The knowledge graph integration provides several benefits:

1. **Enhanced Context Understanding**: The agent can understand the relationships between entities mentioned in conversations.

2. **Semantic Memory**: The agent's memory is structured semantically, allowing for more meaningful retrieval.

3. **Improved Responses**: The agent can use the knowledge graph to provide more relevant and contextual responses.

4. **Cross-Conversation Context**: The agent can leverage information from previous conversations to enhance current responses.

5. **Tool Usage Understanding**: The agent can understand the relationships between tools, their usage, and the entities they operate on.

## Implementation Details

### Entity Extraction

Entities are extracted from text using a language model with a specialized prompt. The model identifies:

- Entity type
- Entity name
- Entity properties

### Relationship Identification

Relationships between entities are identified using a language model with a specialized prompt. The model identifies:

- Source entity
- Target entity
- Relationship type
- Relationship properties

### Graph Storage

The knowledge graph is stored in two forms:

1. **In-Memory Graph**: Using NetworkX for efficient in-memory operations
2. **RDF Graph**: Using RDFLib for semantic queries with SPARQL
3. **Persistent Storage**: Using a database backend for persistence

### Context Retrieval

Context is retrieved from the knowledge graph by:

1. Extracting entities from the user request
2. Finding related entities in the graph
3. Retrieving relationships between entities
4. Providing the relevant context to the agent

## Future Improvements

1. **Improved Entity Extraction**: Enhance entity extraction with specialized models or techniques.

2. **Relationship Inference**: Infer relationships that are not explicitly stated.

3. **Temporal Reasoning**: Add temporal aspects to the knowledge graph.

4. **Multi-Modal Integration**: Integrate with images, audio, and other modalities.

5. **Federated Knowledge Graphs**: Connect with external knowledge graphs.

6. **Reasoning Capabilities**: Add reasoning capabilities on top of the knowledge graph.

7. **User Feedback Integration**: Incorporate user feedback to improve the knowledge graph.

8. **Visualization Tools**: Provide visualization tools for the knowledge graph.

## Conclusion

The knowledge graph integration enhances the agent's context understanding by providing a structured representation of entities and their relationships. This allows the agent to provide more relevant and contextual responses to user requests.
