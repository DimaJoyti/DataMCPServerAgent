"""
Test script for the Research Reports Agent.
This script tests the basic functionality of the research reports agent.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

async def test_model():
    """Test the ChatAnthropic model."""
    try:
        # Initialize model with API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        model = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=api_key)

        # Create messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, can you help me with research on artificial intelligence?")
        ]

        # Invoke the model
        print("Invoking the model...")
        response = await model.ainvoke(messages)

        # Print the response
        print("\nResponse:")
        print(response.content)

        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_model())
