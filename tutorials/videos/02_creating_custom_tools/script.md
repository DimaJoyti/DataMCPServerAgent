# Video Tutorial Script: Creating Custom Tools for DataMCPServerAgent

## Tutorial Title: Creating Custom Tools for DataMCPServerAgent

## Duration: 12-15 minutes

## Prerequisites:

- DataMCPServerAgent installed and configured
- Basic knowledge of Python and object-oriented programming
- Familiarity with LangChain's BaseTool class (optional)

## Learning Objectives:

- Understand the tool architecture in DataMCPServerAgent
- Create a basic custom tool
- Create a more advanced tool with asynchronous support
- Implement a tool provider for dynamic tool creation
- Integrate custom tools with the agent
- Track tool performance

## Outline:

1. Introduction (00:00 - 01:00)
2. Understanding Tool Architecture (01:00 - 03:00)
3. Creating a Basic Custom Tool (03:00 - 05:30)
4. Advanced Tool Development (05:30 - 08:00)
5. Implementing a Tool Provider (08:00 - 10:00)
6. Integrating with the Agent (10:00 - 12:00)
7. Performance Tracking and Feedback (12:00 - 14:00)
8. Conclusion (14:00 - 15:00)

## Script:

### Introduction (00:00 - 01:00)

Hello and welcome to this tutorial on creating custom tools for DataMCPServerAgent. My name is [Your Name], and today I'll be showing you how to extend the agent's capabilities by creating your own custom tools.

Tools are the primary way for the agent to interact with the outside world and perform specific tasks. By creating custom tools, you can give your agent new abilities tailored to your specific needs.

By the end of this tutorial, you'll be able to create basic and advanced custom tools, implement a tool provider, integrate your tools with the agent, and track their performance.

### Understanding Tool Architecture (01:00 - 03:00)

Before we start creating custom tools, let's understand the tool architecture in DataMCPServerAgent.

Tools in DataMCPServerAgent are based on LangChain's `BaseTool` class. Each tool has a name, description, and one or more methods that implement its functionality.

The agent uses the tool's name and description to decide when to use it, and then calls the appropriate method to execute the tool's functionality.

There are three main ways to create tools in DataMCPServerAgent:

1. **Basic Tools**: Simple functions wrapped as tools
2. **Custom Tool Classes**: Classes that extend `BaseTool` with custom functionality
3. **Tool Providers**: Classes that dynamically create and provide tools

Let's look at each of these approaches.

### Creating a Basic Custom Tool (03:00 - 05:30)

Let's start by creating a basic custom tool. We'll create a simple calculator tool that can perform basic arithmetic operations.

First, let's create a new file called `calculator_tool.py` in the `src/tools` directory:

```python
from langchain_core.tools import BaseTool

def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    try:
        # In a real implementation, you would use a safer evaluation method
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Create a tool from the function
calculator_tool = BaseTool(
    name="calculator",
    description="Calculate the result of a mathematical expression",
    func=calculate,
    args_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)
```

This creates a simple calculator tool that can evaluate mathematical expressions. The `args_schema` defines the expected input parameters for the tool.

Now, let's create an asynchronous version of the same tool:

```python
async def calculate_async(expression: str) -> str:
    """Calculate the result of a mathematical expression asynchronously.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    """
    # Same implementation as the synchronous version
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Create an async tool
calculator_async_tool = BaseTool(
    name="calculator_async",
    description="Calculate the result of a mathematical expression asynchronously",
    coroutine=calculate_async,  # Use coroutine instead of func
    args_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)
```

The asynchronous version uses the `coroutine` parameter instead of `func` to specify the asynchronous function.

### Advanced Tool Development (05:30 - 08:00)

Now, let's create a more advanced tool by extending the `BaseTool` class. We'll create a weather tool that can fetch weather information for a location.

```python
from typing import Dict, Any, Optional
from langchain_core.tools import BaseTool

class WeatherTool(BaseTool):
    """Tool for getting weather information."""

    name = "weather"
    description = "Get weather information for a location"

    def _run(self, location: str, units: str = "metric") -> str:
        """Run the weather tool synchronously.

        Args:
            location: Location to get weather for
            units: Units to use (metric or imperial)

        Returns:
            Weather information
        """
        # In a real implementation, this would call a weather API
        # For this example, we'll return mock data

        if units not in ["metric", "imperial"]:
            return "Error: Units must be 'metric' or 'imperial'"

        # Mock weather data
        weather_data = {
            "new york": {
                "metric": {
                    "temperature": 22,
                    "humidity": 65,
                    "wind_speed": 10,
                    "conditions": "Partly cloudy"
                },
                "imperial": {
                    "temperature": 72,
                    "humidity": 65,
                    "wind_speed": 6,
                    "conditions": "Partly cloudy"
                }
            },
            "london": {
                "metric": {
                    "temperature": 18,
                    "humidity": 80,
                    "wind_speed": 15,
                    "conditions": "Rainy"
                },
                "imperial": {
                    "temperature": 64,
                    "humidity": 80,
                    "wind_speed": 9,
                    "conditions": "Rainy"
                }
            }
        }

        # Normalize location
        location_lower = location.lower()

        # Check if we have data for this location
        if location_lower not in weather_data:
            return f"Weather data not available for {location}"

        # Get weather data for the location and units
        data = weather_data[location_lower][units]

        # Format the response
        temp_unit = "°C" if units == "metric" else "°F"
        speed_unit = "km/h" if units == "metric" else "mph"

        response = f"Weather for {location.title()}:\n"
        response += f"Temperature: {data['temperature']}{temp_unit}\n"
        response += f"Humidity: {data['humidity']}%\n"
        response += f"Wind Speed: {data['wind_speed']} {speed_unit}\n"
        response += f"Conditions: {data['conditions']}"

        return response

    async def _arun(self, location: str, units: str = "metric") -> str:
        """Run the weather tool asynchronously.

        Args:
            location: Location to get weather for
            units: Units to use (metric or imperial)

        Returns:
            Weather information
        """
        # For simplicity, we'll just call the synchronous method
        # In a real implementation, you would use an async API client
        return self._run(location, units)
```

This creates a more advanced weather tool with both synchronous and asynchronous methods. The tool can handle different units and provides formatted weather information.

### Implementing a Tool Provider (08:00 - 10:00)

Now, let's create a tool provider that can dynamically create and provide tools. This is useful when you want to create tools based on configuration or runtime conditions.

```python
from typing import Dict, List, Any
from langchain_core.tools import BaseTool

class WeatherToolProvider:
    """Provider for weather tools."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the tool provider.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get("api_key")
        self.default_units = config.get("default_units", "metric")

    async def get_tools(self) -> List[BaseTool]:
        """Get the tools provided by this provider.

        Returns:
            List of tools
        """
        tools = []

        # Create the weather tool
        weather_tool = WeatherTool()

        # Add the tool to the list
        tools.append(weather_tool)

        # If forecast is enabled, create a forecast tool
        if self.config.get("enable_forecast", True):
            forecast_tool = self._create_forecast_tool()
            tools.append(forecast_tool)

        return tools

    def _create_forecast_tool(self) -> BaseTool:
        """Create a forecast tool.

        Returns:
            Forecast tool
        """
        async def get_forecast(location: str, days: int = 5) -> str:
            """Get weather forecast for a location.

            Args:
                location: Location to get forecast for
                days: Number of days to forecast

            Returns:
                Weather forecast
            """
            # In a real implementation, this would call a weather API
            return f"Weather forecast for {location} for the next {days} days:\n" + \
                   f"Day 1: Sunny, 25°C\n" + \
                   f"Day 2: Partly cloudy, 23°C\n" + \
                   f"Day 3: Rainy, 18°C\n" + \
                   f"Day 4: Cloudy, 20°C\n" + \
                   f"Day 5: Sunny, 24°C"

        return BaseTool(
            name="weather_forecast",
            description="Get weather forecast for a location",
            coroutine=get_forecast,
            args_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to get forecast for"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to forecast",
                        "default": 5
                    }
                },
                "required": ["location"]
            }
        )
```

This creates a tool provider that can create and provide weather tools based on configuration. The provider can create a basic weather tool and, if enabled, a forecast tool.

### Integrating with the Agent (10:00 - 12:00)

Now, let's integrate our custom tools with the agent. We'll create a script that configures and runs an agent with our custom tools:

```python
import asyncio
import os
from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from calculator_tool import calculator_tool
from weather_tool import WeatherTool
from weather_tool_provider import WeatherToolProvider

async def run_agent_with_custom_tools():
    """Run the agent with custom tools."""
    # Create custom tools
    calculator = calculator_tool
    weather = WeatherTool()

    # Create tool provider
    provider_config = {
        "api_key": "your-api-key",
        "default_units": "metric",
        "enable_forecast": True
    }
    weather_provider = WeatherToolProvider(provider_config)

    # Configure the agent
    config = {
        "initial_prompt": """
        You are an assistant with custom tools for calculations and weather information.
        You can use the calculator tool to perform mathematical calculations.
        You can use the weather tool to get weather information for a location.
        You can use the weather_forecast tool to get weather forecasts.
        """,
        "additional_tools": [calculator, weather],
        "tool_providers": [weather_provider],
        "verbose": True
    }

    # Run the agent
    await chat_with_advanced_enhanced_agent(config=config)

if __name__ == "__main__":
    asyncio.run(run_agent_with_custom_tools())
```

Save this as `run_agent_with_custom_tools.py` and run it with:

```bash
python run_agent_with_custom_tools.py
```

This will start an agent with our custom calculator and weather tools.

### Performance Tracking and Feedback (12:00 - 14:00)

Finally, let's add performance tracking and feedback to our tools. This helps the agent learn which tools are most effective for different tasks.

```python
from src.memory.memory_persistence import MemoryDatabase
from src.tools.enhanced_tool_selection import ToolPerformanceTracker

async def run_agent_with_performance_tracking():
    """Run the agent with performance tracking."""
    # Initialize dependencies
    db = MemoryDatabase()
    performance_tracker = ToolPerformanceTracker(db)

    # Create custom tools
    calculator = calculator_tool
    weather = WeatherTool()

    # Configure the agent
    config = {
        "initial_prompt": """
        You are an assistant with custom tools for calculations and weather information.
        You can use the calculator tool to perform mathematical calculations.
        You can use the weather tool to get weather information for a location.
        """,
        "additional_tools": [calculator, weather],
        "memory_db": db,
        "performance_tracker": performance_tracker,
        "verbose": True
    }

    # Run the agent
    await chat_with_advanced_enhanced_agent(config=config)

if __name__ == "__main__":
    asyncio.run(run_agent_with_performance_tracking())
```

This adds performance tracking to our agent. The `ToolPerformanceTracker` will track the success rate and execution time of each tool, which helps the agent make better decisions about which tools to use.

### Conclusion (14:00 - 15:00)

In this tutorial, we've covered how to create custom tools for DataMCPServerAgent. We've learned how to create basic tools, advanced tool classes, and tool providers. We've also seen how to integrate these tools with the agent and add performance tracking.

Custom tools are a powerful way to extend the capabilities of your agent and tailor it to your specific needs. By creating your own tools, you can give your agent new abilities and make it more effective at solving your problems.

If you want to learn more, check out our other tutorials on advanced memory management, enhanced tool selection, and building multi-agent systems.

Thanks for watching, and happy coding!

## Visual Notes:

- [00:15] Show slide with tool architecture diagram
- [02:30] Show BaseTool class structure
- [04:15] Highlight the args_schema in the code
- [06:45] Show the WeatherTool class hierarchy
- [09:00] Demonstrate the tool provider creating tools
- [11:00] Show the agent using the custom tools
- [13:00] Show performance metrics for the tools

## Resources:

- [DataMCPServerAgent Tool Development Guide](https://github.com/DimaJoyti/DataMCPServerAgent/blob/main/docs/tool_development.md)
- [LangChain BaseTool Documentation](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- [Example Custom Tools](https://github.com/DimaJoyti/DataMCPServerAgent/tree/main/examples/custom_tool_example.py)
- [Tool Selection Example](https://github.com/DimaJoyti/DataMCPServerAgent/tree/main/examples/tool_selection_example.py)
