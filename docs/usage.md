# Usage Guide

This document provides comprehensive instructions for using the DataMCPServerAgent.

## Running the Agent

### Using the Command Line Interface

The package provides several command-line interfaces for different agent architectures:

- **Basic Agent**:

  ```bash
  datamcpserveragent
  ```

- **Advanced Agent**:

  ```bash
  datamcpserveragent-advanced
  ```

- **Enhanced Agent**:

  ```bash
  datamcpserveragent-enhanced
  ```

- **Advanced Enhanced Agent**:

  ```bash
  datamcpserveragent-advanced-enhanced
  ```

- **Multi-Agent Learning System**:

  ```bash
  datamcpserveragent-multi-agent
  ```

- **Reinforcement Learning Agent**:

  ```bash
  datamcpserveragent-rl
  ```

- **Distributed Memory Agent**:

  ```bash
  datamcpserveragent-distributed
  ```

- **Knowledge Graph Agent**:

  ```bash
  datamcpserveragent-knowledge-graph
  ```

Each CLI command supports additional arguments:

```bash
datamcpserveragent --help
```

Common arguments include:

- `--verbose`: Enable verbose logging
- `--config PATH`: Path to a custom configuration file
- `--memory-backend [redis|mongodb|local]`: Specify the memory backend
- `--model [claude-3-sonnet|claude-3-opus|claude-3-haiku]`: Specify the model to use

### Using the Main Script

You can also use the main script to run the agent with more configuration options:

```bash
python main.py --mode [basic|advanced|enhanced|advanced_enhanced|multi_agent|reinforcement_learning|distributed_memory|knowledge_graph]
```

Additional arguments for the main script:

```bash
python main.py --help
```

Examples:

```bash
# Run the advanced enhanced agent with verbose logging
python main.py --mode advanced_enhanced --verbose

# Run the distributed memory agent with Redis backend
python main.py --mode distributed_memory --memory-backend redis

# Run the reinforcement learning agent with a custom configuration
python main.py --mode reinforcement_learning --config configs/custom_rl_config.json

# Run the knowledge graph agent with a specific model
python main.py --mode knowledge_graph --model claude-3-opus
```

### Using the Python API

You can also use the Python API to run the agent with full customization:

```python
import asyncio
import os
from src.core.main import chat_with_agent
from src.core.advanced_main import chat_with_advanced_agent
from src.core.enhanced_main import chat_with_enhanced_agent
from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.core.multi_agent_main import chat_with_multi_agent_learning_system
from src.core.reinforcement_learning_main import chat_with_rl_agent
from src.core.distributed_memory_main import chat_with_distributed_memory_agent
from src.core.knowledge_graph_main import chat_with_knowledge_graph_agent

# Set environment variables if needed
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
os.environ["BRIGHT_DATA_MCP_KEY"] = "your-mcp-key"

# Basic configuration
config = {
    "verbose": True,
    "memory_backend": "redis",
    "redis_url": "redis://localhost:6379/0",
    "model": "claude-3-sonnet",
    "max_tokens": 4096
}

# Run the basic agent
asyncio.run(chat_with_agent(config=config))

# Run the advanced agent with custom configuration
advanced_config = {
    **config,
    "specialized_agents": ["research", "coding", "creative"],
    "context_window_size": 10
}
asyncio.run(chat_with_advanced_agent(config=advanced_config))

# Run the enhanced agent with learning capabilities
enhanced_config = {
    **config,
    "learning_rate": 0.01,
    "feedback_threshold": 0.7
}
asyncio.run(chat_with_enhanced_agent(config=enhanced_config))

# Run the advanced enhanced agent with all features
advanced_enhanced_config = {
    **config,
    **advanced_config,
    **enhanced_config,
    "tool_selection_strategy": "adaptive"
}
asyncio.run(chat_with_advanced_enhanced_agent(config=advanced_enhanced_config))

# Run the multi-agent learning system
multi_agent_config = {
    **config,
    "num_agents": 3,
    "collaboration_strategy": "consensus"
}
asyncio.run(chat_with_multi_agent_learning_system(config=multi_agent_config))

# Run the reinforcement learning agent
rl_config = {
    **config,
    "reward_model": "user_feedback",
    "exploration_rate": 0.1
}
asyncio.run(chat_with_rl_agent(config=rl_config))

# Run the distributed memory agent
distributed_memory_config = {
    **config,
    "memory_backend": "mongodb",
    "mongodb_uri": "mongodb://localhost:27017/",
    "cache_ttl": 3600
}
asyncio.run(chat_with_distributed_memory_agent(config=distributed_memory_config))

# Run the knowledge graph agent
knowledge_graph_config = {
    **config,
    "graph_backend": "neo4j",
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "password"
}
asyncio.run(chat_with_knowledge_graph_agent(config=knowledge_graph_config))
```

## Special Commands

The agent supports several special commands that can be used during a chat session:

### Basic Agent

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history

### Advanced Agent

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history
- `memory`: View the current memory state
  ```
  memory
  ```
- `memory search <query>`: Search the memory for specific information
  ```
  memory search python examples
  ```
- `agents`: View the available specialized agents
  ```
  agents
  ```
- `switch <agent>`: Switch to a specific specialized agent
  ```
  switch research
  ```

### Enhanced Agent

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history
- `memory`: View the current memory state
- `memory search <query>`: Search the memory for specific information
- `learn`: Trigger learning from feedback
  ```
  learn
  ```
- `insights`: View learning insights
  ```
  insights
  ```
- `feedback <your feedback>`: Provide feedback on the last response
  ```
  feedback This response was very helpful, but could include more examples
  ```
- `preferences`: View and set user preferences
  ```
  preferences
  preferences set response_style detailed
  preferences set code_examples true
  ```

### Advanced Enhanced Agent

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history
- `context`: View the current context
  ```
  context
  ```
- `preferences`: View the current user preferences
  ```
  preferences
  ```
- `preferences set <key> <value>`: Set a user preference
  ```
  preferences set response_style concise
  preferences set code_examples true
  preferences set verbosity high
  ```
- `learn`: Trigger learning from feedback
  ```
  learn
  ```
- `metrics`: View performance metrics
  ```
  metrics
  metrics tools
  metrics memory
  ```
- `feedback <your feedback>`: Provide feedback on the last response
  ```
  feedback The explanation was clear but the code example didn't work
  ```
- `tools`: List available tools
  ```
  tools
  ```
- `tool info <tool_name>`: Get detailed information about a specific tool
  ```
  tool info enhanced_web_search
  ```

### Multi-Agent Learning System

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history
- `knowledge`: View the collaborative knowledge base
  ```
  knowledge
  knowledge search python async
  ```
- `metrics`: View agent performance metrics
  ```
  metrics
  metrics agent research
  ```
- `synergy`: View agent synergy analysis
  ```
  synergy
  ```
- `learn`: Trigger a learning cycle
  ```
  learn
  ```
- `feedback <your feedback>`: Provide feedback on the last response
  ```
  feedback The collaboration between agents produced a comprehensive answer
  ```
- `agents`: List all agents in the system
  ```
  agents
  ```
- `agent <name>`: Get information about a specific agent
  ```
  agent research
  ```
- `collaborate <task>`: Explicitly request collaboration on a task
  ```
  collaborate Find information about Python async programming and provide code examples
  ```

### Reinforcement Learning Agent

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history
- `feedback <your feedback>`: Provide feedback on the last response
  ```
  feedback This response was very helpful and accurate
  ```
- `learn`: Perform batch learning from past interactions
  ```
  learn
  ```
- `policy`: View the current policy
  ```
  policy
  ```
- `rewards`: View the reward history
  ```
  rewards
  ```
- `explore`: Increase exploration rate temporarily
  ```
  explore
  ```

### Distributed Memory Agent

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history
- `feedback <your feedback>`: Provide feedback on the last response
  ```
  feedback The agent remembered our previous conversation accurately
  ```
- `learn`: Perform batch learning from past interactions
  ```
  learn
  ```
- `memory`: View memory summary and statistics
  ```
  memory
  ```
- `memory search <query>`: Search the distributed memory
  ```
  memory search python examples
  ```
- `memory stats`: View memory usage statistics
  ```
  memory stats
  ```
- `cache`: View cache statistics
  ```
  cache
  cache clear
  ```

### Knowledge Graph Agent

- `exit` or `quit`: End the chat session
- `help`: Display available commands
- `clear`: Clear the chat history
- `graph`: View the knowledge graph structure
  ```
  graph
  ```
- `entity <name>`: View information about a specific entity
  ```
  entity Python
  ```
- `relation <type>`: View information about a specific relation type
  ```
  relation depends_on
  ```
- `query <cypher>`: Run a custom Cypher query on the knowledge graph
  ```
  query MATCH (n:Technology)-[:DEPENDS_ON]->(m) RETURN n.name, m.name
  ```

## Tool Usage

DataMCPServerAgent includes several custom tools that can be used during a chat session:

### Enhanced Web Search

```
Search for information about Python async programming
```

### Enhanced Web Scraper

```
Scrape the content from https://example.com and extract the main content
```

### Product Comparison

```
Compare these products:
- https://www.amazon.com/dp/B08N5KWB9H
- https://www.amazon.com/dp/B08N5M7S6K
```

### Social Media Analyzer

```
Analyze this Instagram post: https://www.instagram.com/p/ABC123/
```

## Examples

The `examples/` directory contains example scripts demonstrating different agent architectures and use cases:

### Basic Agent Example

```python
# examples/basic_agent_example.py
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.main import chat_with_agent


async def run_example():
    """Run the basic agent example."""
    print("Running basic agent example...")
    await chat_with_agent()


if __name__ == "__main__":
    asyncio.run(run_example())
```

### Advanced Agent Example

```python
# examples/advanced_agent_example.py
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.advanced_main import chat_with_advanced_agent


async def run_example():
    """Run the advanced agent example."""
    print("Running advanced agent example...")
    
    # Configure the agent
    config = {
        "specialized_agents": ["research", "coding", "creative"],
        "context_window_size": 10,
        "verbose": True
    }
    
    await chat_with_advanced_agent(config=config)


if __name__ == "__main__":
    asyncio.run(run_example())
```

### Product Comparison Example

```python
# examples/product_comparison_example.py
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent


async def run_example():
    """Run the product comparison example."""
    print("Running product comparison example...")
    
    # Configure the agent with product comparison focus
    config = {
        "initial_prompt": """
        You are a product comparison assistant. You can help users compare products
        by analyzing their features, prices, reviews, and more. You can use the
        product_comparison tool to compare products from various e-commerce sites.
        """,
        "verbose": True
    }
    
    await chat_with_advanced_enhanced_agent(config=config)


if __name__ == "__main__":
    asyncio.run(run_example())
```

### Social Media Analysis Example

```python
# examples/social_media_analysis_example.py
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent


async def run_example():
    """Run the social media analysis example."""
    print("Running social media analysis example...")
    
    # Configure the agent with social media analysis focus
    config = {
        "initial_prompt": """
        You are a social media analysis assistant. You can help users analyze
        social media content from platforms like Instagram, Facebook, and Twitter/X.
        You can use the social_media_analyzer tool to analyze posts, profiles,
        and engagement metrics.
        """,
        "verbose": True
    }
    
    await chat_with_advanced_enhanced_agent(config=config)


if __name__ == "__main__":
    asyncio.run(run_example())
```

## Troubleshooting

### Common Issues

#### API Key Issues

If you encounter API key errors:

```
Error: Invalid API key or API key not found
```

Make sure you have set the required API keys in your `.env` file or environment variables:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
BRIGHT_DATA_MCP_KEY=your_bright_data_mcp_key
```

#### Memory Backend Issues

If you encounter memory backend errors:

```
Error: Failed to connect to memory backend
```

Make sure your memory backend (Redis, MongoDB) is running and accessible:

```bash
# Check Redis
redis-cli ping

# Check MongoDB
mongosh --eval "db.adminCommand('ping')"
```

#### Tool Execution Issues

If you encounter tool execution errors:

```
Error: Failed to execute tool
```

Check the tool's requirements and make sure all dependencies are installed:

```bash
python install_dependencies.py --all
```

### Getting Help

If you encounter issues not covered in this guide, you can:

1. Check the logs in the `logs/` directory
2. Run the agent with the `--verbose` flag for more detailed logging
3. Check the GitHub repository for known issues and solutions
4. Contact the maintainers for support
