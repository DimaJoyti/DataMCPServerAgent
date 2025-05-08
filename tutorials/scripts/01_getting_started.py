"""
Tutorial: Getting Started with DataMCPServerAgent

This tutorial demonstrates how to set up and use the basic agent in DataMCPServerAgent.

Learning objectives:
- Install and configure DataMCPServerAgent
- Run the basic agent
- Interact with the agent
- Use special commands

Prerequisites:
- Python 3.8 or higher installed
- Anthropic API key
- Bright Data MCP credentials
"""

import asyncio
import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
from src.core.main import chat_with_agent

# Check if required environment variables are set
def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ["ANTHROPIC_API_KEY", "BRIGHT_DATA_MCP_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("Error: The following environment variables are not set:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    return True

# Simulate user input
async def simulate_user_input(prompt, delay=1.0):
    """Simulate user input with a delay."""
    print(f"\nUser: {prompt}")
    time.sleep(delay)
    
    # In a real tutorial, this would call the agent
    # For this example, we'll simulate the agent's response
    if prompt.lower() == "hello":
        return "Hello! I'm the DataMCPServerAgent. How can I help you today?"
    elif prompt.lower() == "what can you do?":
        return "I can help you with various tasks, including searching the web, analyzing data, and more. I have access to several tools that allow me to perform these tasks."
    elif prompt.lower() == "help":
        return """
Available commands:
- exit or quit: End the chat session
- help: Display this help message
- clear: Clear the chat history
- memory: View the current memory state
"""
    elif prompt.lower() == "memory":
        return """
Current memory state:
- User asked: "hello"
- User asked: "what can you do?"
- User asked: "help"
"""
    elif prompt.lower() in ["exit", "quit"]:
        return "Goodbye! Thank you for using the DataMCPServerAgent."
    else:
        return f"I received your message: '{prompt}'. How can I assist you further?"

async def run_tutorial():
    """Run the tutorial."""
    print("Starting tutorial: Getting Started with DataMCPServerAgent")
    
    # Step 1: Check environment variables
    print("\nStep 1: Checking environment variables")
    if not check_environment_variables():
        print("\nPlease set the required environment variables and try again.")
        return
    
    print("Environment variables are set correctly!")
    
    # Step 2: Configure the agent
    print("\nStep 2: Configuring the agent")
    config = {
        "verbose": True,  # Enable verbose logging
        "memory_backend": "local",  # Use local memory backend
        "model": "claude-3-sonnet",  # Use Claude 3 Sonnet model
        "max_tokens": 4096  # Maximum number of tokens to generate
    }
    
    print("Agent configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 3: Simulate running the agent
    print("\nStep 3: Running the agent")
    print("\nIn a real tutorial, you would run the agent with:")
    print("  python main.py --mode basic")
    print("Or using the Python API:")
    print("  asyncio.run(chat_with_agent(config=config))")
    
    print("\nFor this tutorial, we'll simulate the agent's behavior.")
    
    # Step 4: Interact with the agent
    print("\nStep 4: Interacting with the agent")
    print("\nLet's simulate some interactions with the agent:")
    
    # Simulate a conversation
    prompts = [
        "hello",
        "what can you do?",
        "help",
        "memory",
        "exit"
    ]
    
    for prompt in prompts:
        response = await simulate_user_input(prompt)
        print(f"Agent: {response}")
        time.sleep(1.0)  # Add a delay between interactions
    
    # Step 5: Explain how to customize the agent
    print("\nStep 5: Customizing the agent")
    print("\nYou can customize the agent by modifying the configuration:")
    print("  - Change the model (claude-3-sonnet, claude-3-opus, claude-3-haiku)")
    print("  - Change the memory backend (local, redis, mongodb)")
    print("  - Enable or disable verbose logging")
    print("  - Adjust the maximum number of tokens")
    
    print("\nExample of running the agent with custom configuration:")
    print("  python main.py --mode basic --model claude-3-opus --verbose")
    
    print("\nTutorial completed!")
    print("\nNext steps:")
    print("  1. Try running the actual agent with your own configuration")
    print("  2. Explore the advanced agent features in the next tutorial")
    print("  3. Learn how to create custom tools for the agent")

if __name__ == "__main__":
    asyncio.run(run_tutorial())