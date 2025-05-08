# DataMCPServerAgent Documentation

Welcome to the DataMCPServerAgent documentation. This documentation provides comprehensive information about the DataMCPServerAgent project, including installation instructions, usage guides, and architecture details.

## Overview

DataMCPServerAgent is a sophisticated agent system built on top of Bright Data MCP. It provides advanced agent architectures with memory persistence, tool selection, and learning capabilities.

## Documentation Sections

- [Installation Guide](installation.md): Instructions for installing the DataMCPServerAgent
- [Usage Guide](usage.md): Instructions for using the DataMCPServerAgent
- [Architecture](architecture.md): Overview of the DataMCPServerAgent architecture
- [Memory Systems](memory.md): Detailed information about memory systems
- [Distributed Memory](distributed_memory.md): Detailed information about the distributed memory capabilities
- [Multi-Agent Learning](multi_agent_learning.md): Detailed information about the multi-agent learning system
- [Reinforcement Learning](reinforcement_learning.md): Detailed information about the reinforcement learning capabilities

## Agent Architectures

The project implements several agent architectures with increasing levels of sophistication:

1. **Basic Agent**: Simple ReAct agent with Bright Data MCP tools
2. **Advanced Agent**: Agent with specialized sub-agents, tool selection, and memory
3. **Enhanced Agent**: Agent with memory persistence, enhanced tool selection, and learning capabilities
4. **Advanced Enhanced Agent**: Agent with context-aware memory, adaptive learning, and sophisticated tool selection
5. **Multi-Agent Learning System**: System with collaborative learning, knowledge sharing, and performance optimization between multiple agents
6. **Reinforcement Learning Agent**: Agent that learns from rewards and improves through experience
7. **Distributed Memory Agent**: Agent with scalable distributed memory across Redis and MongoDB backends

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DimaJoyti/DataMCPServerAgent.git
cd DataMCPServerAgent

# Install the package
pip install -e .

# Create .env file from template
cp .env.template .env
# Edit .env with your credentials
```

### Running the Agent

```bash
# Run the basic agent
python main.py --mode basic

# Run the advanced agent
python main.py --mode advanced

# Run the enhanced agent
python main.py --mode enhanced

# Run the advanced enhanced agent
python main.py --mode advanced_enhanced

# Run the multi-agent learning system
python main.py --mode multi_agent

# Run the reinforcement learning agent
python main.py --mode reinforcement_learning

# Run the distributed memory agent
python main.py --mode distributed_memory
```

## Project Structure

The project is organized into the following directories:

- `src/` - Main source code directory

  - `core/` - Core functionality and entry points
  - `agents/` - Agent-related modules
  - `memory/` - Memory-related modules
  - `tools/` - Tool-related modules
  - `utils/` - Utility functions

- `docs/` - Documentation files
- `examples/` - Example scripts
- `tests/` - Test files

## Contributing

Contributions to the DataMCPServerAgent project are welcome! Please see the [Contributing Guide](contributing.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
