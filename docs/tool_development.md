# Tool Development Guide

This guide provides instructions for extending the DataMCPServerAgent with new custom tools.

## Table of Contents

- [Tool Development Guide](#tool-development-guide)
  - [Table of Contents](#table-of-contents)
  - [Tool Development Overview](#tool-development-overview)
  - [Creating a Basic Tool](#creating-a-basic-tool)
  - [Creating a Custom Tool Class](#creating-a-custom-tool-class)
  - [Tool Registration](#tool-registration)
  - [Tool Performance Tracking](#tool-performance-tracking)
  - [Best Practices](#best-practices)
  - [Advanced Tool Development](#advanced-tool-development)
    - [Composite Tools](#composite-tools)
    - [Tool with Memory](#tool-with-memory)
  - [Testing Tools](#testing-tools)

## Tool Development Overview

DataMCPServerAgent uses LangChain's `BaseTool` class as the foundation for all tools. Tools are functions or methods that can be invoked by the agent to perform specific tasks.

The agent architecture supports several types of tools:

1. **Basic Tools**: Simple functions wrapped as tools
2. **Custom Tools**: Enhanced tools with specialized functionality
3. **Composite Tools**: Tools that combine multiple other tools

## Creating a Basic Tool

The simplest way to create a tool is to use LangChain's `BaseTool` class:

```python
from langchain_core.tools import BaseTool

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

# Create a tool from the function
sum_tool = BaseTool(
    name="calculate_sum",
    description="Calculate the sum of two numbers",
    func=calculate_sum,
    args_schema={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"}
        },
        "required": ["a", "b"]
    }
)
```

For asynchronous functions, you can use the `coroutine` parameter instead of `func`:

```python
async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    # Async implementation
    return "Data from URL"

# Create an async tool
fetch_tool = BaseTool(
    name="fetch_data",
    description="Fetch data from a URL",
    coroutine=fetch_data,
    args_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch data from"}
        },
        "required": ["url"]
    }
)
```

## Creating a Custom Tool Class

For more complex tools, you can create a custom tool class by extending `BaseTool`:

```python
from typing import Dict, Any
from langchain_core.tools import BaseTool

class DataAnalysisTool(BaseTool):
    """Tool for analyzing data."""

    name = "data_analysis"
    description = "Analyze data and provide insights"

    def _run(self, data_source: str, analysis_type: str = "basic") -> str:
        """Run the tool synchronously."""
        # Implementation
        return f"Analysis of {data_source} using {analysis_type} analysis"

    async def _arun(self, data_source: str, analysis_type: str = "basic") -> str:
        """Run the tool asynchronously."""
        # Async implementation
        return f"Analysis of {data_source} using {analysis_type} analysis"
```

## Tool Registration

To make your tools available to the agent, you need to register them with the agent's tool registry:

```python
from src.core.agent_factory import AgentFactory

# Create your tools
my_tools = [sum_tool, fetch_tool, DataAnalysisTool()]

# Create an agent with your tools
agent_factory = AgentFactory()
agent = agent_factory.create_agent(
    agent_type="advanced",
    additional_tools=my_tools
)
```

Alternatively, you can create a tool provider class that dynamically creates and provides tools:

```python
from typing import List
from langchain_core.tools import BaseTool

class MyToolProvider:
    """Provider for custom tools."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the tool provider."""
        self.config = config

    async def get_tools(self) -> List[BaseTool]:
        """Get the tools provided by this provider."""
        tools = []

        # Create and add tools based on configuration
        if self.config.get("enable_sum_tool", True):
            tools.append(sum_tool)

        if self.config.get("enable_fetch_tool", True):
            tools.append(fetch_tool)

        if self.config.get("enable_analysis_tool", True):
            tools.append(DataAnalysisTool())

        return tools
```

Then register the tool provider with the agent factory:

```python
from src.core.agent_factory import AgentFactory

# Create your tool provider
tool_provider = MyToolProvider(config={"enable_analysis_tool": True})

# Create an agent with your tool provider
agent_factory = AgentFactory()
agent = agent_factory.create_agent(
    agent_type="advanced",
    tool_providers=[tool_provider]
)
```

## Tool Performance Tracking

DataMCPServerAgent includes a tool performance tracking system that monitors tool usage and success rates. You can integrate your tools with this system to track their performance:

```python
from src.tools.enhanced_tool_selection import ToolPerformanceTracker
from src.memory.memory_persistence import MemoryDatabase

# Initialize the performance tracker
db = MemoryDatabase()
performance_tracker = ToolPerformanceTracker(db)

# Track tool execution
performance_tracker.start_execution("my_tool")

try:
    # Execute the tool
    result = my_tool.invoke(args)
    success = True
except Exception:
    success = False

# Record the execution result
execution_time = performance_tracker.end_execution("my_tool", success)
```

## Best Practices

When developing tools for DataMCPServerAgent, follow these best practices:

1. **Clear Names and Descriptions**: Use clear, descriptive names and descriptions for your tools to help the agent understand when to use them.

2. **Proper Error Handling**: Handle errors gracefully and provide informative error messages.

3. **Input Validation**: Validate inputs before processing to prevent errors and security issues.

4. **Asynchronous Support**: Implement both synchronous (`_run`) and asynchronous (`_arun`) methods for better performance.

5. **Documentation**: Document your tools thoroughly, including parameters, return values, and examples.

6. **Testing**: Write tests for your tools to ensure they work correctly.

7. **Performance Optimization**: Optimize your tools for performance, especially for frequently used operations.

8. **Security Considerations**: Be mindful of security implications, especially for tools that access external resources.

## Advanced Tool Development

### Composite Tools

Composite tools combine multiple tools to perform complex tasks:

```python
from langchain_core.tools import BaseTool

class CompositeSearchTool(BaseTool):
    """Tool that combines web search and content analysis."""

    name = "composite_search"
    description = "Search the web and analyze the results"

    def __init__(self, search_tool: BaseTool, analysis_tool: BaseTool):
        """Initialize the composite tool."""
        self.search_tool = search_tool
        self.analysis_tool = analysis_tool
        super().__init__()

    async def _arun(self, query: str) -> str:
        """Run the composite tool."""
        # First, search the web
        search_results = await self.search_tool.ainvoke({"query": query})

        # Then, analyze the results
        analysis = await self.analysis_tool.ainvoke({
            "data_source": search_results,
            "analysis_type": "comprehensive"
        })

        return f"Search Results:\n{search_results}\n\nAnalysis:\n{analysis}"
```

### Tool with Memory

Tools can also integrate with the agent's memory system:

```python
from langchain_core.tools import BaseTool
from src.memory.memory_persistence import MemoryDatabase

class MemoryAwareSearchTool(BaseTool):
    """Search tool that uses memory to improve results."""

    name = "memory_aware_search"
    description = "Search the web with memory of past searches"

    def __init__(self, search_tool: BaseTool, memory_db: MemoryDatabase):
        """Initialize the memory-aware search tool."""
        self.search_tool = search_tool
        self.memory_db = memory_db
        super().__init__()

    async def _arun(self, query: str) -> str:
        """Run the memory-aware search tool."""
        # Check if we have relevant past searches
        past_searches = self.memory_db.get_related_memories(
            "search_results",
            query,
            limit=3
        )

        # If we have relevant past searches, use them to enhance the query
        if past_searches:
            enhanced_query = f"{query} (excluding: {', '.join([p['query'] for p in past_searches])})"
        else:
            enhanced_query = query

        # Perform the search
        results = await self.search_tool.ainvoke({"query": enhanced_query})

        # Store the results in memory
        self.memory_db.save_memory(
            "search_results",
            {
                "query": query,
                "results": results
            }
        )

        return results
```

## Testing Tools

It's important to test your tools to ensure they work correctly. Here's an example of how to test a tool:

```python
import pytest
from src.tools.my_tools import MyCustomTool

@pytest.fixture
def my_tool():
    """Create a test instance of MyCustomTool."""
    return MyCustomTool()

def test_my_tool_basic(my_tool):
    """Test basic functionality of MyCustomTool."""
    result = my_tool.invoke({"param1": "value1", "param2": "value2"})
    assert "expected output" in result

@pytest.mark.asyncio
async def test_my_tool_async(my_tool):
    """Test async functionality of MyCustomTool."""
    result = await my_tool.ainvoke({"param1": "value1", "param2": "value2"})
    assert "expected output" in result

def test_my_tool_error_handling(my_tool):
    """Test error handling of MyCustomTool."""
    with pytest.raises(ValueError):
        my_tool.invoke({"param1": "invalid value"})
```

Save your tests in the `tests/tools/` directory and run them with pytest:

```bash
pytest tests/tools/test_my_tools.py
```
