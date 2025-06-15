# Usage Guide

This guide provides practical examples and tutorials for using DataMCPServerAgent effectively. From basic agent interactions to advanced multi-agent workflows.

## 🚀 Quick Start

### Starting the System

```bash
# Start API server
python app/main_consolidated.py api

# Start CLI interface
python app/main_consolidated.py cli

# Start web UI (in separate terminal)
cd agent-ui && npm run dev
```

### Your First Agent

```python
import asyncio
from app.domain.services.agent_service import AgentService

async def create_first_agent():
    # Initialize agent service
    agent_service = AgentService()
    
    # Create a research agent
    agent = await agent_service.create_agent(
        agent_type="research",
        name="My Research Assistant",
        configuration={
            "max_iterations": 10,
            "enable_learning": True,
            "memory_backend": "postgresql"
        }
    )
    
    print(f"Created agent: {agent.agent_id}")
    
    # Execute a simple task
    result = await agent.execute_task(
        "Find information about renewable energy trends in 2024"
    )
    
    print(f"Result: {result.content}")
    return result

# Run the example
asyncio.run(create_first_agent())
```

## 🌐 Using the Web Interface

### Accessing the Dashboard

1. Open your browser to `http://localhost:3000`
2. Navigate to the Agents section
3. Click "Create New Agent"
4. Fill in the agent details:
   - **Name**: "Research Assistant"
   - **Type**: "Research"
   - **Configuration**: Default settings

### Chat Interface

1. Click on your created agent
2. Start a conversation in the chat interface
3. Try these example queries:
   - "Research the latest AI developments"
   - "Find market data for renewable energy stocks"
   - "Summarize recent papers on machine learning"

### Monitoring Performance

1. Go to the Analytics tab
2. View real-time metrics:
   - Response times
   - Success rates
   - Memory usage
   - Task completion rates

## 🤖 Agent Types and Use Cases

### Research Agents

Perfect for information gathering and analysis:

```python
research_agent = await agent_service.create_agent(
    agent_type="research",
    name="Market Research Assistant",
    configuration={
        "search_depth": "comprehensive",
        "source_types": ["academic", "news", "reports"],
        "max_sources": 50
    }
)

# Use cases:
await research_agent.execute_task("Research competitors in the AI space")
await research_agent.execute_task("Find latest trends in sustainable technology")
await research_agent.execute_task("Analyze market size for electric vehicles")
```

### Trading Agents

For financial analysis and algorithmic trading:

```python
trading_agent = await agent_service.create_agent(
    agent_type="trading",
    name="Crypto Trading Bot",
    configuration={
        "strategy": "momentum",
        "risk_tolerance": "medium",
        "max_position_size": 1000
    }
)

# Use cases:
await trading_agent.execute_task("Analyze BTC/USD trend")
await trading_agent.execute_task("Generate trading signals for ETH")
await trading_agent.execute_task("Calculate portfolio risk metrics")
```

### Brand Agents

For customer service and marketing:

```python
brand_agent = await agent_service.create_agent(
    agent_type="brand",
    name="Customer Support Bot",
    configuration={
        "personality": "friendly_professional",
        "knowledge_base": "company_docs",
        "escalation_enabled": True
    }
)

# Use cases:
await brand_agent.execute_task("Handle customer inquiry about product features")
await brand_agent.execute_task("Generate marketing copy for new product")
await brand_agent.execute_task("Respond to social media comments")
```

### Semantic Agents

For knowledge extraction and reasoning:

```python
semantic_agent = await agent_service.create_agent(
    agent_type="semantic",
    name="Knowledge Extractor",
    configuration={
        "knowledge_graph_enabled": True,
        "entity_extraction": True,
        "reasoning_depth": "deep"
    }
)

# Use cases:
await semantic_agent.execute_task("Extract entities from research papers")
await semantic_agent.execute_task("Build knowledge graph from documents")
await semantic_agent.execute_task("Answer complex reasoning questions")
```

## 🔗 Multi-Agent Workflows

### Coordinated Research

```python
from app.application.orchestration import MultiAgentOrchestrator

async def coordinated_research():
    orchestrator = MultiAgentOrchestrator()
    
    # Add specialized agents
    search_agent = await orchestrator.add_agent("research", "search_specialist")
    analysis_agent = await orchestrator.add_agent("research", "data_analyst")
    summary_agent = await orchestrator.add_agent("research", "content_summarizer")
    
    # Define workflow
    workflow = {
        "name": "Market Research Workflow",
        "steps": [
            {
                "agent": "search_specialist",
                "task": "Find information about AI startups in 2024",
                "output": "raw_data"
            },
            {
                "agent": "data_analyst", 
                "task": "Analyze the collected data for trends",
                "input": "raw_data",
                "output": "analysis"
            },
            {
                "agent": "content_summarizer",
                "task": "Create executive summary",
                "input": "analysis",
                "output": "final_report"
            }
        ]
    }
    
    # Execute workflow
    result = await orchestrator.execute_workflow(workflow)
    return result

# Run coordinated research
result = asyncio.run(coordinated_research())
print(result["final_report"])
```

### Trading Strategy Development

```python
async def trading_strategy_workflow():
    orchestrator = MultiAgentOrchestrator()
    
    # Market analysis team
    market_agent = await orchestrator.add_agent("trading", "market_analyzer")
    risk_agent = await orchestrator.add_agent("trading", "risk_manager")
    execution_agent = await orchestrator.add_agent("trading", "trade_executor")
    
    # Parallel analysis
    market_analysis = await market_agent.execute_task("Analyze BTC market conditions")
    risk_assessment = await risk_agent.execute_task("Calculate portfolio risk")
    
    # Generate trading decision
    if market_analysis.confidence > 0.7 and risk_assessment.risk_level < 0.5:
        trade_result = await execution_agent.execute_task(
            f"Execute trade based on analysis: {market_analysis.recommendation}"
        )
        return trade_result
    
    return {"decision": "Hold", "reason": "Market conditions not optimal"}

# Execute trading workflow
trade_decision = asyncio.run(trading_strategy_workflow())
```

## 📡 API Usage Examples

### REST API

#### Create Agent

```bash
curl -X POST http://localhost:8003/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "research",
    "name": "API Research Agent",
    "configuration": {
      "max_iterations": 5,
      "enable_learning": true
    }
  }'
```

#### Execute Task

```bash
curl -X POST http://localhost:8003/api/v1/agents/{agent_id}/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Research latest developments in quantum computing",
    "priority": 1,
    "timeout": 300
  }'
```

#### Get Task Status

```bash
curl http://localhost:8003/api/v1/agents/{agent_id}/tasks/{task_id}
```

### WebSocket API

```javascript
// Connect to agent WebSocket
const ws = new WebSocket('ws://localhost:8003/ws/agents/{agent_id}');

// Handle incoming messages
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'task_update':
            console.log('Task progress:', data.progress);
            break;
        case 'task_complete':
            console.log('Task completed:', data.result);
            break;
        case 'error':
            console.error('Error:', data.message);
            break;
    }
};

// Send task
ws.send(JSON.stringify({
    type: 'execute_task',
    data: {
        description: 'Analyze market trends',
        priority: 1
    }
}));
```

### Python SDK

```python
from datamcp_sdk import DataMCPClient

# Initialize client
client = DataMCPClient(
    api_url="http://localhost:8003",
    api_key="your-api-key"  # if authentication enabled
)

# Create agent
agent = await client.agents.create(
    agent_type="research",
    name="SDK Agent",
    configuration={"max_iterations": 10}
)

# Execute task
task = await client.agents.execute_task(
    agent_id=agent.agent_id,
    description="Research renewable energy market",
    wait_for_completion=True
)

print(f"Task result: {task.result}")
```

## 🧠 Advanced Features

### Reinforcement Learning

```python
# Enable RL for agent optimization
rl_agent = await agent_service.create_agent(
    agent_type="research",
    name="Learning Research Agent",
    configuration={
        "reinforcement_learning": {
            "enabled": True,
            "algorithm": "dqn",
            "learning_rate": 0.001,
            "exploration_rate": 0.1
        }
    }
)

# The agent will learn and improve over time
for i in range(100):
    result = await rl_agent.execute_task(f"Research topic {i}")
    # Agent learns from feedback and improves performance
```

### Memory and Context

```python
# Create agent with enhanced memory
memory_agent = await agent_service.create_agent(
    agent_type="research",
    name="Memory Enhanced Agent",
    configuration={
        "memory": {
            "type": "semantic",
            "capacity": 10000,
            "recall_threshold": 0.8
        },
        "context_window": 50
    }
)

# Agent remembers previous interactions
await memory_agent.execute_task("Research AI companies")
await memory_agent.execute_task("Compare the AI companies from before with traditional tech companies")
# Agent uses memory from first task in second task
```

### Tool Integration

```python
# Create agent with custom tools
tool_agent = await agent_service.create_agent(
    agent_type="research",
    name="Tool Enhanced Agent",
    configuration={
        "tools": [
            "web_search",
            "document_analysis", 
            "data_visualization",
            "email_sender"
        ]
    }
)

# Agent can use multiple tools
await tool_agent.execute_task("Research market data and email me a visualization")
```

## 📊 Monitoring and Analytics

### Performance Metrics

```python
# Get agent performance metrics
metrics = await agent_service.get_agent_metrics(agent_id)

print(f"Success rate: {metrics.success_rate}%")
print(f"Average response time: {metrics.avg_response_time}s")
print(f"Tasks completed: {metrics.total_tasks}")
print(f"Learning progress: {metrics.learning_score}")
```

### Real-time Monitoring

```python
# Monitor agent performance in real-time
async def monitor_agent(agent_id):
    async for update in agent_service.stream_metrics(agent_id):
        print(f"Current task: {update.current_task}")
        print(f"Progress: {update.progress}%")
        print(f"ETA: {update.estimated_completion}")

# Start monitoring
asyncio.create_task(monitor_agent(agent.agent_id))
```

## 🔧 Configuration Best Practices

### Environment-Specific Configurations

```python
# Development configuration
dev_config = {
    "debug": True,
    "log_level": "DEBUG",
    "max_iterations": 5,
    "timeout": 60
}

# Production configuration
prod_config = {
    "debug": False,
    "log_level": "INFO", 
    "max_iterations": 20,
    "timeout": 300,
    "caching_enabled": True,
    "monitoring_enabled": True
}
```

### Security Configuration

```python
# Secure agent configuration
secure_agent = await agent_service.create_agent(
    agent_type="research",
    name="Secure Agent",
    configuration={
        "security": {
            "input_validation": True,
            "output_filtering": True,
            "rate_limiting": True,
            "audit_logging": True
        }
    }
)
```

## 🎯 Common Patterns

### Error Handling

```python
from app.core.exceptions import AgentError, TaskTimeoutError

async def robust_task_execution():
    try:
        result = await agent.execute_task("Complex research task", timeout=300)
        return result
    except TaskTimeoutError:
        # Handle timeout
        print("Task timed out, trying with simpler approach")
        return await agent.execute_task("Simplified research task", timeout=60)
    except AgentError as e:
        # Handle agent-specific errors
        print(f"Agent error: {e.message}")
        return None
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        return None
```

### Batch Processing

```python
async def batch_process_tasks():
    tasks = [
        "Research company A",
        "Research company B", 
        "Research company C"
    ]
    
    # Process tasks concurrently
    results = await asyncio.gather(*[
        agent.execute_task(task) for task in tasks
    ])
    
    return results

# Execute batch processing
batch_results = asyncio.run(batch_process_tasks())
```

### Progressive Enhancement

```python
async def progressive_research(topic):
    # Start with basic research
    basic_result = await agent.execute_task(f"Basic research on {topic}")
    
    # Enhance with detailed analysis if basic research was successful
    if basic_result.confidence > 0.7:
        detailed_result = await agent.execute_task(
            f"Detailed analysis of {topic} based on: {basic_result.summary}"
        )
        return detailed_result
    
    return basic_result
```

## 📚 Next Steps

After mastering the basics:

1. **Explore Advanced Features**: Try reinforcement learning and multi-agent coordination
2. **Build Custom Tools**: Create specialized tools for your use case
3. **Deploy to Production**: Follow the [Deployment Guide](deployment_guide.md)
4. **Join the Community**: Share your experiences and learn from others

## 💡 Tips and Tricks

### Performance Optimization

- Use specific agent types for specialized tasks
- Configure appropriate timeouts for different task complexities
- Enable caching for repetitive operations
- Monitor memory usage with large datasets

### Best Practices

- Always handle errors gracefully
- Use descriptive names for agents and tasks
- Monitor agent performance regularly
- Keep configurations environment-specific
- Test thoroughly before production deployment

---

**Happy building!** 🚀 You're now ready to create powerful AI agent applications with DataMCPServerAgent.
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

### Research Assistant Example

```python
# examples/research_assistant_example.py
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.research_assistant import run_research_assistant


def run_example():
    """Run the research assistant example."""
    print("Running research assistant example...")

    # Run the research assistant
    run_research_assistant()


if __name__ == "__main__":
    run_example()
```

## Troubleshooting

### Common Issues

#### API Key Issues

If you encounter API key errors:

```bash
Error: Invalid API key or API key not found
```

Make sure you have set the required API keys in your `.env` file or environment variables:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
BRIGHT_DATA_MCP_KEY=your_bright_data_mcp_key
```

#### Memory Backend Issues

If you encounter memory backend errors:

```bash
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

```bash
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
