"""
Example of creating and using a custom tool with DataMCPServerAgent.
"""

import asyncio
import os
import sys
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from mcp import ClientSession

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.memory.memory_persistence import MemoryDatabase
from src.tools.enhanced_tool_selection import ToolPerformanceTracker


class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    name = "weather_tool"
    description = "Get weather information for a location"
    
    async def _arun(self, location: str, units: str = "metric") -> str:
        """Run the weather tool asynchronously.
        
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
            },
            "tokyo": {
                "metric": {
                    "temperature": 28,
                    "humidity": 70,
                    "wind_speed": 8,
                    "conditions": "Sunny"
                },
                "imperial": {
                    "temperature": 82,
                    "humidity": 70,
                    "wind_speed": 5,
                    "conditions": "Sunny"
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
        
        response = f"## Weather for {location.title()}\n\n"
        response += f"**Temperature**: {data['temperature']}{temp_unit}\n"
        response += f"**Humidity**: {data['humidity']}%\n"
        response += f"**Wind Speed**: {data['wind_speed']} {speed_unit}\n"
        response += f"**Conditions**: {data['conditions']}\n"
        
        return response


class CurrencyConverterTool(BaseTool):
    """Tool for converting currencies."""
    
    name = "currency_converter"
    description = "Convert an amount from one currency to another"
    
    async def _arun(self, amount: float, from_currency: str, to_currency: str) -> str:
        """Run the currency converter tool asynchronously.
        
        Args:
            amount: Amount to convert
            from_currency: Currency to convert from
            to_currency: Currency to convert to
            
        Returns:
            Conversion result
        """
        # In a real implementation, this would call a currency API
        # For this example, we'll use mock exchange rates
        
        # Mock exchange rates (relative to USD)
        exchange_rates = {
            "usd": 1.0,
            "eur": 0.85,
            "gbp": 0.75,
            "jpy": 110.0,
            "cad": 1.25,
            "aud": 1.35,
            "cny": 6.45
        }
        
        # Normalize currencies
        from_currency_lower = from_currency.lower()
        to_currency_lower = to_currency.lower()
        
        # Check if we have exchange rates for these currencies
        if from_currency_lower not in exchange_rates:
            return f"Exchange rate not available for {from_currency}"
            
        if to_currency_lower not in exchange_rates:
            return f"Exchange rate not available for {to_currency}"
            
        # Convert to USD first (if not already USD)
        usd_amount = amount / exchange_rates[from_currency_lower]
        
        # Convert from USD to target currency
        converted_amount = usd_amount * exchange_rates[to_currency_lower]
        
        # Format the response
        response = f"## Currency Conversion\n\n"
        response += f"{amount} {from_currency.upper()} = {converted_amount:.2f} {to_currency.upper()}\n\n"
        response += f"Exchange rate: 1 {from_currency.upper()} = {exchange_rates[to_currency_lower] / exchange_rates[from_currency_lower]:.4f} {to_currency.upper()}"
        
        return response


class CustomToolProvider:
    """Provider for custom tools."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the tool provider.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    async def get_tools(self) -> List[BaseTool]:
        """Get the tools provided by this provider.
        
        Returns:
            List of tools
        """
        tools = []
        
        # Add weather tool if enabled
        if self.config.get("enable_weather_tool", True):
            tools.append(WeatherTool())
        
        # Add currency converter tool if enabled
        if self.config.get("enable_currency_converter", True):
            tools.append(CurrencyConverterTool())
        
        return tools


async def run_example():
    """Run the custom tool example."""
    print("Running custom tool example...")
    
    # Create custom tools
    weather_tool = WeatherTool()
    currency_converter = CurrencyConverterTool()
    
    # Configure the agent with custom tools
    config = {
        "initial_prompt": """
        You are an assistant with access to custom tools for weather information and currency conversion.
        You can use the weather_tool to get weather information for a location.
        You can use the currency_converter to convert between currencies.
        """,
        "additional_tools": [weather_tool, currency_converter],
        "verbose": True
    }
    
    # Run the agent
    await chat_with_advanced_enhanced_agent(config=config)


async def run_example_with_provider():
    """Run the custom tool example using a tool provider."""
    print("Running custom tool example with provider...")
    
    # Configure the tool provider
    provider_config = {
        "enable_weather_tool": True,
        "enable_currency_converter": True
    }
    
    # Create the tool provider
    tool_provider = CustomToolProvider(provider_config)
    
    # Configure the agent with the tool provider
    config = {
        "initial_prompt": """
        You are an assistant with access to custom tools for weather information and currency conversion.
        You can use the weather_tool to get weather information for a location.
        You can use the currency_converter to convert between currencies.
        """,
        "tool_providers": [tool_provider],
        "verbose": True
    }
    
    # Run the agent
    await chat_with_advanced_enhanced_agent(config=config)


async def run_example_with_performance_tracking():
    """Run the custom tool example with performance tracking."""
    print("Running custom tool example with performance tracking...")
    
    # Initialize dependencies
    db = MemoryDatabase()
    performance_tracker = ToolPerformanceTracker(db)
    
    # Create custom tools
    weather_tool = WeatherTool()
    currency_converter = CurrencyConverterTool()
    
    # Configure the agent with custom tools and performance tracking
    config = {
        "initial_prompt": """
        You are an assistant with access to custom tools for weather information and currency conversion.
        You can use the weather_tool to get weather information for a location.
        You can use the currency_converter to convert between currencies.
        """,
        "additional_tools": [weather_tool, currency_converter],
        "memory_db": db,
        "performance_tracker": performance_tracker,
        "verbose": True
    }
    
    # Run the agent
    await chat_with_advanced_enhanced_agent(config=config)


if __name__ == "__main__":
    # Choose which example to run
    example_type = "basic"  # Change to "provider" or "performance" for other examples
    
    if example_type == "basic":
        asyncio.run(run_example())
    elif example_type == "provider":
        asyncio.run(run_example_with_provider())
    elif example_type == "performance":
        asyncio.run(run_example_with_performance_tracking())
    else:
        print(f"Unknown example type: {example_type}")