"""
Example of using the enhanced tool selection capabilities of DataMCPServerAgent.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import BaseTool
from langchain_anthropic import ChatAnthropic

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.memory.memory_persistence import MemoryDatabase
from src.tools.enhanced_tool_selection import EnhancedToolSelector, ToolPerformanceTracker

class SearchTool(BaseTool):
    """Tool for searching the web."""

    name = "search"
    description = "Search the web for information"

    async def _arun(self, query: str) -> str:
        """Run the search tool asynchronously.

        Args:
            query: Search query

        Returns:
            Search results
        """
        # Mock search results
        return f"## Search Results for '{query}'\n\n" + \
               f"1. Result 1 for {query}\n" + \
               f"2. Result 2 for {query}\n" + \
               f"3. Result 3 for {query}\n"

class CalculatorTool(BaseTool):
    """Tool for performing calculations."""

    name = "calculator"
    description = "Perform mathematical calculations"

    async def _arun(self, expression: str) -> str:
        """Run the calculator tool asynchronously.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Calculation result
        """
        try:
            # Evaluate the expression (in a real implementation, use a safer method)
            result = eval(expression)
            return f"## Calculation Result\n\n{expression} = {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

class WeatherTool(BaseTool):
    """Tool for getting weather information."""

    name = "weather"
    description = "Get weather information for a location"

    async def _arun(self, location: str) -> str:
        """Run the weather tool asynchronously.

        Args:
            location: Location to get weather for

        Returns:
            Weather information
        """
        # Mock weather data
        return f"## Weather for {location}\n\n" + \
               f"Temperature: 22°C\n" + \
               f"Humidity: 65%\n" + \
               f"Conditions: Partly cloudy\n"

class TranslationTool(BaseTool):
    """Tool for translating text."""

    name = "translate"
    description = "Translate text from one language to another"

    async def _arun(self, text: str, source_lang: str, target_lang: str) -> str:
        """Run the translation tool asynchronously.

        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language

        Returns:
            Translated text
        """
        # Mock translation
        return f"## Translation from {source_lang} to {target_lang}\n\n" + \
               f"Original: {text}\n" + \
               f"Translated: [Translated version of '{text}' in {target_lang}]\n"

async def demonstrate_tool_selection():
    """Demonstrate the enhanced tool selection process."""
    print("Demonstrating enhanced tool selection...")

    # Create tools
    tools = [
        SearchTool(),
        CalculatorTool(),
        WeatherTool(),
        TranslationTool()
    ]

    # Initialize dependencies
    model = ChatAnthropic(model="claude-3-sonnet-20240229")
    db = MemoryDatabase()
    performance_tracker = ToolPerformanceTracker(db)

    # Initialize the tool selector
    tool_selector = EnhancedToolSelector(
        model=model,
        tools=tools,
        db=db,
        performance_tracker=performance_tracker
    )

    # Example requests
    requests = [
        "What is the capital of France?",
        "Calculate 125 * 37",
        "What's the weather like in Tokyo?",
        "Translate 'Hello, how are you?' from English to Spanish",
        "What is the population of New York City?",
        "What is the square root of 144?"
    ]

    # Process each request
    for request in requests:
        print(f"\nRequest: {request}")

        # Select tools for the request
        selection = await tool_selector.select_tools(request)

        print(f"Selected tools: {selection['selected_tools']}")
        print(f"Reasoning: {selection['reasoning']}")
        print(f"Execution order: {selection['execution_order']}")
        print(f"Fallback tools: {selection['fallback_tools']}")

        # Simulate tool execution
        if selection['selected_tools']:
            tool_name = selection['selected_tools'][0]
            tool = next((t for t in tools if t.name == tool_name), None)

            if tool:
                print(f"Executing tool: {tool_name}")

                # Start tracking execution
                performance_tracker.start_execution(tool_name)

                try:
                    # Simulate tool execution with appropriate arguments
                    if tool_name == "search":
                        result = await tool.ainvoke({"query": request})
                        success = True
                    elif tool_name == "calculator":
                        # Extract expression from the request
                        expression = request.replace("Calculate ", "").strip()
                        result = await tool.ainvoke({"expression": expression})
                        success = True
                    elif tool_name == "weather":
                        # Extract location from the request
                        location = request.replace("What's the weather like in ", "").replace("?", "").strip()
                        result = await tool.ainvoke({"location": location})
                        success = True
                    elif tool_name == "translate":
                        # For simplicity, use fixed arguments
                        result = await tool.ainvoke({
                            "text": "Hello, how are you?",
                            "source_lang": "English",
                            "target_lang": "Spanish"
                        })
                        success = True
                    else:
                        result = "Unknown tool"
                        success = False

                    print(f"Result: {result}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                    success = False

                # End tracking execution
                execution_time = performance_tracker.end_execution(tool_name, success)
                print(f"Execution time: {execution_time:.2f} seconds")

                # Get feedback on the execution
                feedback = await tool_selector.provide_execution_feedback(
                    request=request,
                    tool_name=tool_name,
                    tool_args={},  # Simplified for the example
                    tool_result=result,
                    execution_time=execution_time,
                    success=success
                )

                print(f"Feedback: {feedback}")

        print("-" * 50)

async def run_example():
    """Run the tool selection example."""
    await demonstrate_tool_selection()

async def run_agent_with_enhanced_selection():
    """Run the agent with enhanced tool selection."""
    print("Running agent with enhanced tool selection...")

    # Create tools
    tools = [
        SearchTool(),
        CalculatorTool(),
        WeatherTool(),
        TranslationTool()
    ]

    # Initialize dependencies
    db = MemoryDatabase()
    performance_tracker = ToolPerformanceTracker(db)

    # Configure the agent
    config = {
        "initial_prompt": """
        You are an assistant with access to various tools.
        You can use the search tool to find information on the web.
        You can use the calculator tool to perform mathematical calculations.
        You can use the weather tool to get weather information for a location.
        You can use the translate tool to translate text between languages.

        The system will automatically select the most appropriate tools for each request.
        """,
        "additional_tools": tools,
        "memory_db": db,
        "performance_tracker": performance_tracker,
        "tool_selection_strategy": "enhanced",
        "verbose": True
    }

    # Run the agent
    await chat_with_advanced_enhanced_agent(config=config)

if __name__ == "__main__":
    # Choose which example to run
    example_type = "demonstration"  # Change to "agent" to run the agent example

    if example_type == "demonstration":
        asyncio.run(run_example())
    elif example_type == "agent":
        asyncio.run(run_agent_with_enhanced_selection())
    else:
        print(f"Unknown example type: {example_type}")
