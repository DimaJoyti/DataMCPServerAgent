# DataMCPServerAgent

A sophisticated Python-based agent system that combines context-aware memory, adaptive learning, and enhanced tool selection capabilities. Built on top of Bright Data's MCP (Model Context Protocol) server.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Context-Aware Memory**: Maintains and utilizes contextual information across interactions
- **Adaptive Learning**: Learns from user interactions and feedback to improve responses
- **Enhanced Tool Selection**: Sophisticated tool selection and performance tracking
- **Multi-Agent Learning**: Collaborative learning capabilities across multiple agent instances
- **Distributed Memory**: Scalable memory persistence across multiple storage backends
- **Bright Data Integration**: Seamless integration with Bright Data's web unlocker and proxy services

## Prerequisites

- Python 3.8 or higher
- Node.js (for Bright Data MCP)
- Bright Data MCP credentials
- Anthropic API key (for Claude model)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DimaJoyti/DataMCPServerAgent.git
cd DataMCPServerAgent
```

2. Install dependencies:

```bash
python install_dependencies.py
```

3. Set up environment variables:

```bash
cp .env.example .env
```

Then edit `.env` with your credentials.

## Usage

Basic usage:

```python
from src.core.main import chat_with_agent

# Start the agent
asyncio.run(chat_with_agent())
```

For advanced features:

```python
from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent

# Start the advanced enhanced agent
asyncio.run(chat_with_advanced_enhanced_agent())
```

See the `examples/` directory for more usage examples.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Installation Guide](docs/installation.md)
- [Architecture Overview](docs/architecture.md)
- [Memory Management](docs/memory.md)
- [Multi-Agent Learning](docs/multi_agent_learning.md)
- [Usage Guide](docs/usage.md)

## Contributing

See [Contributing Guide](docs/contributing.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Bright Data for their MCP server implementation
- Anthropic for the Claude model API
- The LangChain community for various tools and utilities

## Contact

- GitHub: [@DimaJoyti](https://github.com/DimaJoyti)
