# 🤖 DataMCPServerAgent v2.0

> **Advanced AI Agent System with MCP Integration**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

DataMCPServerAgent is a production-ready, enterprise-grade AI agent system built with modern Python practices. It provides a comprehensive platform for building, deploying, and managing AI agents with advanced capabilities including memory persistence, learning, and multi-modal interactions.

## ✨ Key Features

### 🧠 **Advanced AI Capabilities**
- **Multi-Agent Coordination**: Sophisticated agent orchestration and collaboration
- **Adaptive Learning**: Continuous learning from interactions and feedback
- **Context-Aware Memory**: Persistent, searchable memory with intelligent retrieval
- **Tool Integration**: Extensible tool system with 50+ built-in tools
- **Multi-Modal Support**: Text, voice, and visual interaction capabilities

### 🏗️ **Enterprise Architecture**
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **Type Safety**: Full type hints with mypy validation
- **Async/Await**: High-performance asynchronous operations
- **Microservices Ready**: Containerized and Kubernetes-native
- **Observability**: Comprehensive logging, metrics, and tracing

### 🔧 **Developer Experience**
- **Single Command Setup**: Get started in under 5 minutes
- **Hot Reload**: Instant feedback during development
- **Rich CLI**: Beautiful command-line interface with Typer
- **API-First**: OpenAPI documentation and SDK generation
- **Testing**: 90%+ test coverage with pytest

### 🚀 **Production Ready**
- **Scalable**: Horizontal scaling with load balancing
- **Secure**: JWT authentication, rate limiting, CORS protection
- **Reliable**: Circuit breakers, retries, and graceful degradation
- **Monitored**: Prometheus metrics and health checks
- **Deployed**: Docker, Kubernetes, and cloud-native

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DimaJoyti/DataMCPServerAgent.git
cd DataMCPServerAgent
```

2. **Install dependencies**
```bash
# Using pip
pip install -r requirements.txt

# Using uv (recommended)
uv pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Start the application**
```bash
# API Server
python app/main_improved.py api

# CLI Interface
python app/main_improved.py cli

# Background Worker
python app/main_improved.py worker
```

### Docker Quick Start

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
open http://localhost:8002/docs
```

## 📖 Usage Examples

### API Server
```bash
# Start development server
python app/main_improved.py api --reload --log-level DEBUG

# Start production server
python app/main_improved.py api --workers 4 --env production
```

### CLI Interface
```python
# Interactive mode
python app/main_improved.py cli --interactive

# Batch processing
echo "Analyze this data" | python app/main_improved.py cli --interactive=false
```

### Python SDK
```python
from app.agents import create_agent
from app.tools import get_available_tools

# Create an agent
agent = await create_agent(
    name="data-analyst",
    capabilities=["data_analysis", "visualization"],
    tools=get_available_tools("data")
)

# Execute a task
result = await agent.execute(
    "Analyze the sales data and create a summary report"
)

print(result.summary)
```

## 🏗️ Architecture

DataMCPServerAgent follows Clean Architecture principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   FastAPI   │  │     CLI     │  │   WebRTC    │        │
│  │     API     │  │ Interface   │  │   Calls     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Use Cases  │  │  Commands   │  │   Queries   │        │
│  │ Orchestrate │  │   Modify    │  │    Read     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Domain Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Agents    │  │    Tasks    │  │    Users    │        │
│  │ Aggregates  │  │ Aggregates  │  │ Aggregates  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Database   │  │    Cache    │  │  External   │        │
│  │ PostgreSQL  │  │    Redis    │  │   APIs      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **Agents**: Autonomous AI entities with specialized capabilities
- **Tasks**: Work units with lifecycle management and progress tracking
- **Tools**: Extensible functionality modules (data, communication, analysis)
- **Memory**: Persistent, context-aware storage with intelligent retrieval
- **Communication**: Multi-modal interaction (text, voice, video)

## 🔧 Configuration

DataMCPServerAgent uses a hierarchical configuration system:

```python
# Environment variables
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379

# Configuration file (.env)
API_HOST=0.0.0.0
API_PORT=8002
LOG_LEVEL=INFO

# Runtime configuration
python app/main_improved.py api --host 0.0.0.0 --port 8002
```

### Configuration Sections

- **Application**: Basic app settings and metadata
- **Database**: Connection, pooling, and migration settings
- **Cache**: Redis configuration and caching strategies
- **Security**: Authentication, authorization, and encryption
- **Monitoring**: Logging, metrics, and health checks
- **Integrations**: External services (Cloudflare, email, WebRTC)

## 🧪 Testing

```bash
# Run all tests
python app/main_improved.py test

# Run with coverage
python app/main_improved.py test --coverage

# Run specific tests
python app/main_improved.py test --pattern "test_agents"

# Performance tests
pytest tests/performance/ -v
```

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Full workflow testing
- **Performance Tests**: Load and stress testing

## 📊 Monitoring

### Health Checks
```bash
# System status
python app/main_improved.py status

# API health
curl http://localhost:8002/health
```

### Metrics
- **Application Metrics**: Request rates, response times, error rates
- **Business Metrics**: Agent performance, task completion rates
- **Infrastructure Metrics**: CPU, memory, database connections
- **Custom Metrics**: Domain-specific measurements

### Observability Stack
- **Logging**: Structured JSON logs with correlation IDs
- **Metrics**: Prometheus with Grafana dashboards
- **Tracing**: Distributed tracing with Jaeger
- **Alerting**: PagerDuty integration for critical issues

## 🚀 Deployment

### Local Development
```bash
# Development server
python app/main_improved.py api --reload

# With Docker
docker-compose up --build
```

### Production Deployment
```bash
# Docker
docker build -t datamcp-agent .
docker run -p 8002:8002 datamcp-agent

# Kubernetes
kubectl apply -f deployment/kubernetes/
```

### Cloud Platforms
- **AWS**: ECS, EKS, Lambda
- **Google Cloud**: GKE, Cloud Run
- **Azure**: AKS, Container Instances
- **Cloudflare**: Workers, Pages, R2

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run quality checks
black app/
mypy app/
ruff check app/
```

## 📚 Documentation

- **[API Reference](docs/api/)** - Complete API documentation
- **[User Guide](docs/guides/)** - Step-by-step tutorials
- **[Architecture](docs/architecture/)** - System design and patterns
- **[Deployment](docs/deployment/)** - Production deployment guides
- **[Development](docs/development/)** - Developer resources

## 🔗 Links

- **Documentation**: [https://datamcp.dev/docs](https://datamcp.dev/docs)
- **API Reference**: [https://datamcp.dev/api](https://datamcp.dev/api)
- **GitHub**: [https://github.com/DimaJoyti/DataMCPServerAgent](https://github.com/DimaJoyti/DataMCPServerAgent)
- **Discord**: [https://discord.gg/datamcp](https://discord.gg/datamcp)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Anthropic** for Claude AI model
- **Bright Data** for MCP server implementation
- **FastAPI** for the excellent web framework
- **Pydantic** for data validation
- **The Open Source Community** for amazing tools and libraries

---

<div align="center">
  <strong>Built with ❤️ by the DataMCPServerAgent team</strong>
</div>
