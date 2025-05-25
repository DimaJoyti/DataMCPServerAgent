# DataMCPServerAgent

A sophisticated Python-based agent system that combines context-aware memory, adaptive learning, and enhanced tool selection capabilities. Built on top of Bright Data's MCP (Model Context Protocol) server.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Context-Aware Memory**: Maintains and utilizes contextual information across interactions
- **Adaptive Learning**: Learns from user interactions and feedback to improve responses
- **Enhanced Tool Selection**: Sophisticated tool selection and performance tracking
- **Multi-Agent Learning**: Collaborative learning capabilities across multiple agent instances
- **Reinforcement Learning**: Continuous improvement through reward-based learning
- **Distributed Memory**: Scalable memory persistence across Redis and MongoDB backends with caching
- **Knowledge Graph Integration**: Enhanced context understanding through entity and relationship modeling
- **Enhanced Error Recovery**: Sophisticated retry strategies, automatic fallback mechanisms, and self-healing capabilities
- **Advanced Error Analysis**: Error clustering, root cause analysis, correlation analysis, and predictive error detection
- **Bright Data Integration**: Seamless integration with Bright Data's web unlocker and proxy services

## 🏗️ Data Pipeline System

### Enterprise Data Processing Infrastructure

- **Pipeline Orchestration**: Advanced workflow management with dependency resolution
- **Data Ingestion**: Batch and streaming data ingestion from multiple sources
- **Data Transformation**: ETL/ELT pipelines with validation and quality checks
- **Processing Engines**: Parallel batch processing and real-time stream processing
- **Monitoring & Observability**: Comprehensive metrics and monitoring
- **Storage Integration**: Unified access to databases, object storage, and file systems

### Supported Data Sources

- **Databases**: PostgreSQL, MySQL, SQLite, MongoDB
- **Files**: CSV, JSON, Parquet, Excel
- **APIs**: REST APIs with authentication and pagination
- **Object Storage**: S3-compatible storage (AWS S3, MinIO)
- **Streaming**: Apache Kafka, Redis Streams

### Processing Capabilities

- **Batch Processing**: Large-scale data processing with parallel execution
- **Stream Processing**: Real-time data processing with windowing
- **Data Validation**: Schema validation and quality checks
- **Error Handling**: Retry mechanisms and error recovery
- **Scheduling**: Cron-based pipeline scheduling

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

For reinforcement learning:

```python
from src.core.reinforcement_learning_main import chat_with_rl_agent

# Start the reinforcement learning agent
asyncio.run(chat_with_rl_agent())
```

For distributed memory:

```python
from src.core.distributed_memory_main import chat_with_distributed_memory_agent

# Start the distributed memory agent
asyncio.run(chat_with_distributed_memory_agent())
```

For knowledge graph integration:

```python
from src.core.knowledge_graph_main import chat_with_knowledge_graph_agent

# Start the knowledge graph agent
asyncio.run(chat_with_knowledge_graph_agent())
```

For enhanced error recovery:

```python
from src.core.error_recovery_main import chat_with_error_recovery_agent

# Start the error recovery agent
asyncio.run(chat_with_error_recovery_agent())
```

For data pipeline processing:

```python
from src.core.data_pipeline_main import chat_with_data_pipeline_system

# Start the data pipeline system
asyncio.run(chat_with_data_pipeline_system())
```

Or run the data pipeline example:

```python
# Run the comprehensive data pipeline example
python examples/data_pipeline_example.py
```

See the `examples/` directory for more usage examples.

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### 🏗️ Architecture & Design
- [System Architecture Blueprint](docs/system_architecture_blueprint.md) - Comprehensive system architecture overview
- [Component Specifications](docs/component_specifications.md) - Detailed component interfaces and specifications
- [Data Flow & Integration](docs/data_flow_integration.md) - Data flow patterns and integration architecture
- [Architecture Overview](docs/architecture.md) - High-level system architecture

### 🚀 Deployment & Operations
- [Deployment & Operations Guide](docs/deployment_operations.md) - Complete deployment and operational procedures
- [API Reference](docs/api_reference.md) - Comprehensive REST API and SDK documentation
- [Installation Guide](docs/installation.md) - Setup and installation instructions

### 🔧 Feature Guides
- [Data Pipeline Guide](docs/data_pipeline_guide.md) - Enterprise data processing capabilities
- [Memory Management](docs/memory.md) - Memory system overview
- [Distributed Memory](docs/distributed_memory.md) - Distributed memory architecture
- [Knowledge Graph](docs/knowledge_graph.md) - Knowledge graph integration
- [Multi-Agent Learning](docs/multi_agent_learning.md) - Multi-agent coordination
- [Reinforcement Learning](docs/reinforcement_learning.md) - RL capabilities
- [Reinforcement Learning Memory Persistence](docs/reinforcement_learning_memory.md) - RL memory systems
- [Error Recovery](docs/error_recovery.md) - Error handling systems
- [Advanced Error Analysis](docs/advanced_error_analysis.md) - Advanced error analysis

### 💻 Development & Usage
- [Usage Guide](docs/usage.md) - Getting started guide
- [Custom Tools](docs/custom_tools.md) - Building custom tools and integrations
- [Tool Development Guide](docs/tool_development.md) - Development best practices
- [Contributing Guide](docs/contributing.md) - How to contribute to the project

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
