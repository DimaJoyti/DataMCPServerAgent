# Usage Guide

This document provides instructions for using the DataMCPServerAgent.

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

### Using the Main Script

You can also use the main script to run the agent:

```bash
python main.py --mode [basic|advanced|enhanced|advanced_enhanced|multi_agent|reinforcement_learning]
```

For example, to run the advanced enhanced agent:

```bash
python main.py --mode advanced_enhanced
```

Or to run the multi-agent learning system:

```bash
python main.py --mode multi_agent
```

### Using the Python API

You can also use the Python API to run the agent:

```python
import asyncio
from src.core.main import chat_with_agent
from src.core.advanced_main import chat_with_advanced_agent
from src.core.enhanced_main import chat_with_enhanced_agent
from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.core.multi_agent_main import chat_with_multi_agent_learning_system
from src.core.reinforcement_learning_main import chat_with_rl_agent

# Run the basic agent
asyncio.run(chat_with_agent())

# Run the advanced agent
asyncio.run(chat_with_advanced_agent())

# Run the enhanced agent
asyncio.run(chat_with_enhanced_agent())

# Run the advanced enhanced agent
asyncio.run(chat_with_advanced_enhanced_agent())

# Run the multi-agent learning system
asyncio.run(chat_with_multi_agent_learning_system())

# Run the reinforcement learning agent
asyncio.run(chat_with_rl_agent())
```

## Special Commands

The agent supports several special commands:

### Basic Agent

- `exit` or `quit`: End the chat

### Advanced Agent

- `exit` or `quit`: End the chat
- `memory`: View the current memory state
- `agents`: View the available specialized agents

### Enhanced Agent

- `exit` or `quit`: End the chat
- `memory`: View the current memory state
- `learn`: Trigger learning from feedback
- `insights`: View learning insights
- `feedback <your feedback>`: Provide feedback on the last response

### Advanced Enhanced Agent

- `exit` or `quit`: End the chat
- `context`: View the current context
- `preferences`: View the current user preferences
- `learn`: Trigger learning from feedback
- `metrics`: View performance metrics
- `feedback <your feedback>`: Provide feedback on the last response

### Multi-Agent Learning System

- `exit` or `quit`: End the chat
- `knowledge`: View the collaborative knowledge base
- `metrics`: View agent performance metrics
- `synergy`: View agent synergy analysis
- `learn`: Trigger a learning cycle
- `feedback <your feedback>`: Provide feedback on the last response

### Reinforcement Learning Agent

- `exit` or `quit`: End the chat
- `feedback: <your feedback>`: Provide feedback on the last response
- `learn`: Perform batch learning from past interactions

## Examples

See the `examples/` directory for example scripts demonstrating different agent architectures:

- `examples/basic_example.py`: Basic agent example
- `examples/advanced_example.py`: Advanced agent example
- `examples/enhanced_capabilities_example.py`: Enhanced agent example
- `examples/advanced_enhanced_example.py`: Advanced enhanced agent example
- `examples/multi_agent_learning_example.py`: Multi-agent learning system example
- `examples/reinforcement_learning_example.py`: Reinforcement learning agent example
- `examples/product_comparison_example.py`: Product comparison example
- `examples/social_media_analysis_example.py`: Social media analysis example
