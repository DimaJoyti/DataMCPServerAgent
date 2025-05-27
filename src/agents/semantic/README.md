# Semantic Agents System

## 🧠 Overview

The Semantic Agents System is an advanced AI agent architecture that provides semantic understanding, inter-agent communication, and scalable performance for generative AI solutions. This system represents a significant enhancement to the DataMCPServerAgent platform.

## ✨ Key Features

### 🎯 Semantic Understanding
- **Intent Recognition**: Advanced natural language understanding
- **Context Management**: Semantic context preservation across interactions
- **Entity Extraction**: Automatic identification of key entities and relationships
- **Knowledge Integration**: Integration with knowledge graphs and memory systems

### 🤝 Inter-Agent Communication
- **Message Bus**: Reliable message passing between agents
- **Topic Subscriptions**: Event-driven communication patterns
- **Request-Response**: Synchronous communication for task coordination
- **Broadcasting**: Efficient one-to-many communication

### 📈 Scalability & Performance
- **Auto-Scaling**: Automatic scaling based on system metrics
- **Load Balancing**: Intelligent task distribution across agents
- **Performance Monitoring**: Real-time performance tracking and optimization
- **Caching**: Intelligent caching for improved response times

### 🔧 Specialized Agents
- **Data Analysis Agent**: Statistical analysis, visualization, pattern recognition
- **Document Processing Agent**: Parsing, summarization, entity extraction
- **Knowledge Extraction Agent**: Concept extraction, knowledge graph building
- **Reasoning Agent**: Logical inference, problem decomposition
- **Search Agent**: Semantic search, query expansion, result ranking

## 🚀 Quick Start

### Installation

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Install Pylint for code quality
uv pip install pylint>=3.0.0
```

### Running the System

```bash
# Start semantic agents system
python app/main_improved.py semantic-agents

# With custom configuration
python app/main_improved.py semantic-agents --api-port 8003 --log-level DEBUG
```

### Basic Usage

```python
from src.agents.semantic.main import SemanticAgentsSystem

# Create and initialize system
system = SemanticAgentsSystem()
await system.initialize()

# Execute a task
result = await system.coordinator.execute_task(
    task_description="Analyze sales data and identify trends",
    required_capabilities=["statistical_analysis", "trend_analysis"]
)

print(result)
```

## 📚 API Reference

### Task Execution

```http
POST /semantic-agents/tasks/execute
Content-Type: application/json

{
  "task_description": "Analyze customer feedback sentiment",
  "agent_type": "document_processing",
  "required_capabilities": ["sentiment_analysis"],
  "priority": 5,
  "collaborative": false
}
```

### Agent Management

```http
# List all agents
GET /semantic-agents/agents

# Get specific agent
GET /semantic-agents/agents/{agent_id}

# Create new agent
POST /semantic-agents/agents
{
  "agent_type": "data_analysis",
  "name": "custom_data_agent",
  "capabilities": ["statistical_analysis", "visualization"]
}

# Delete agent
DELETE /semantic-agents/agents/{agent_id}
```

### System Monitoring

```http
# System status
GET /semantic-agents/system/status

# Performance bottlenecks
GET /semantic-agents/performance/bottlenecks

# Cache statistics
GET /semantic-agents/cache/stats
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Semantic Coordinator                     │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Task Distribution & Management              │
│  └─────────────────────────────────────────────────────────┤
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Data      │  │  Document   │  │ Knowledge   │         │
│  │  Analysis   │  │ Processing  │  │ Extraction  │   ...   │
│  │   Agent     │  │   Agent     │  │   Agent     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                 Communication Hub                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Message Bus │  │ Topic Mgmt  │  │ Event Queue │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Performance Tracker │ Auto Scaler │ Load Balancer         │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Agent Types

### Data Analysis Agent
**Capabilities**: Statistical analysis, data visualization, pattern recognition, trend analysis

```python
# Example usage
result = await coordinator.execute_task(
    task_description="Calculate correlation between variables",
    agent_type="data_analysis",
    required_capabilities=["statistical_analysis"]
)
```

### Document Processing Agent
**Capabilities**: Document parsing, content summarization, entity extraction, classification

```python
# Example usage
result = await coordinator.execute_task(
    task_description="Summarize research paper",
    agent_type="document_processing",
    required_capabilities=["content_summarization"]
)
```

### Knowledge Extraction Agent
**Capabilities**: Concept extraction, relationship identification, knowledge graph construction

```python
# Example usage
result = await coordinator.execute_task(
    task_description="Extract concepts from technical documentation",
    agent_type="knowledge_extraction",
    required_capabilities=["concept_extraction"]
)
```

### Reasoning Agent
**Capabilities**: Logical inference, causal reasoning, problem decomposition

```python
# Example usage
result = await coordinator.execute_task(
    task_description="Analyze cause-effect relationships",
    agent_type="reasoning",
    required_capabilities=["causal_reasoning"]
)
```

### Search Agent
**Capabilities**: Semantic search, query expansion, result ranking

```python
# Example usage
result = await coordinator.execute_task(
    task_description="Find relevant documents",
    agent_type="search",
    required_capabilities=["semantic_search"]
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your_api_key_here

# Optional
LOG_LEVEL=INFO
MEMORY_DB_PATH=./data/memory.db
KNOWLEDGE_GRAPH_DB_PATH=./data/knowledge_graph.db
CACHE_MAX_SIZE=1000
PERFORMANCE_TRACKING_ENABLED=true
AUTO_SCALING_ENABLED=true
```

### Agent Configuration

```python
from src.agents.semantic.base_semantic_agent import SemanticAgentConfig

config = SemanticAgentConfig(
    name="custom_agent",
    specialization="custom_domain",
    capabilities=["capability_1", "capability_2"],
    model_name="claude-3-sonnet-20240229",
    temperature=0.1,
    max_tokens=4000,
    memory_enabled=True,
    knowledge_graph_enabled=True,
)
```

## 📊 Performance Monitoring

### Metrics Tracked
- **Operation Duration**: Response times for all operations
- **Success Rates**: Success/failure rates per agent
- **Resource Usage**: CPU, memory, and disk usage
- **Cache Performance**: Hit/miss ratios and cache efficiency
- **Load Distribution**: Task distribution across agents

### Bottleneck Detection
The system automatically identifies:
- Slow operations and performance bottlenecks
- High failure rates and error patterns
- Resource constraints and scaling needs
- Cache inefficiencies

### Optimization Recommendations
Automatic recommendations for:
- Performance optimization strategies
- Resource allocation improvements
- Scaling decisions
- Cache configuration tuning

## 🧪 Testing

### Running Tests

```bash
# Run all semantic agent tests
pytest tests/test_semantic_agents.py -v

# Run with coverage
pytest tests/test_semantic_agents.py --cov=src/agents/semantic --cov-report=html

# Run specific test categories
pytest tests/test_semantic_agents.py::TestBaseSemanticAgent -v
```

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing
- **Performance Tests**: Load and stress testing
- **API Tests**: REST API endpoint testing

## 🔍 Code Quality

### Pylint Integration
```bash
# Run Pylint on semantic agents
pylint src/agents/semantic/

# With configuration
pylint --rcfile=pyproject.toml src/agents/semantic/
```

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Quality Metrics
- **Pylint Score**: Target 9.0+/10.0
- **MyPy Coverage**: 95%+ type coverage
- **Test Coverage**: 90%+ code coverage
- **Cyclomatic Complexity**: <10 per function

## 🚨 Troubleshooting

### Common Issues

1. **Agent Not Responding**
   ```bash
   # Check agent status
   curl http://localhost:8003/semantic-agents/agents/{agent_id}
   
   # Check system logs
   tail -f logs/semantic_agents.log
   ```

2. **Performance Issues**
   ```bash
   # Check bottlenecks
   curl http://localhost:8003/semantic-agents/performance/bottlenecks
   
   # Monitor system resources
   curl http://localhost:8003/semantic-agents/system/status
   ```

3. **Memory Issues**
   ```bash
   # Clear cache
   curl -X POST http://localhost:8003/semantic-agents/cache/clear
   
   # Check cache stats
   curl http://localhost:8003/semantic-agents/cache/stats
   ```

### Debug Mode
```bash
# Enable debug logging
python app/main_improved.py semantic-agents --log-level DEBUG

# Check detailed logs
tail -f logs/semantic_agents_debug.log
```

## 🔮 Future Enhancements

- **Multi-modal Capabilities**: Support for text, image, and audio processing
- **Federated Learning**: Distributed learning across agent instances
- **Advanced Reasoning**: Integration with external knowledge bases
- **Real-time Collaboration**: Enhanced collaborative task execution
- **Custom Agent Templates**: Plugin system for custom agent types
- **Workflow Orchestration**: Complex multi-step workflow management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../../LICENSE) file for details.

## 🤝 Contributing

Please read [CONTRIBUTING.md](../../../CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the [documentation](../../../docs/SEMANTIC_AGENTS_GUIDE.md)
- Review the [troubleshooting guide](../../../docs/SEMANTIC_AGENTS_GUIDE.md#troubleshooting)
