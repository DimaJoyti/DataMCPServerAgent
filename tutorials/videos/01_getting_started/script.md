# Video Tutorial Script: Getting Started with DataMCPServerAgent

## Tutorial Title: Getting Started with DataMCPServerAgent

## Duration: 10-12 minutes

## Prerequisites:
- Python 3.8 or higher installed
- Basic knowledge of Python
- Anthropic API key
- Bright Data MCP credentials

## Learning Objectives:
- Understand what DataMCPServerAgent is and its key features
- Install and configure DataMCPServerAgent
- Run your first agent
- Understand the basic commands and interactions
- Learn how to customize the agent configuration

## Outline:
1. Introduction (00:00 - 01:00)
2. Installation and Setup (01:00 - 03:30)
3. Running Your First Agent (03:30 - 06:00)
4. Basic Commands and Interactions (06:00 - 08:30)
5. Customizing the Agent (08:30 - 10:30)
6. Conclusion (10:30 - 11:30)

## Script:

### Introduction (00:00 - 01:00)

Hello and welcome to this tutorial on getting started with DataMCPServerAgent. My name is [Your Name], and today I'll be showing you how to set up and use this powerful agent system.

DataMCPServerAgent is a sophisticated Python-based agent system that combines context-aware memory, adaptive learning, and enhanced tool selection capabilities. It's built on top of Bright Data's MCP (Model Context Protocol) server and provides a flexible framework for creating intelligent agents.

By the end of this tutorial, you'll be able to install DataMCPServerAgent, run your first agent, understand the basic commands, and customize the agent configuration.

### Installation and Setup (01:00 - 03:30)

Let's start by installing DataMCPServerAgent. First, you'll need to clone the repository from GitHub:

```bash
git clone https://github.com/DimaJoyti/DataMCPServerAgent.git
cd DataMCPServerAgent
```

Next, you'll need to install the dependencies. The project includes a convenient script for this:

```bash
python install_dependencies.py
```

This will install all the required packages, including LangChain, Anthropic's Claude client, and other dependencies.

Now, let's set up the environment variables. The project includes a template file that you can copy:

```bash
cp .env.template .env
```

Open the `.env` file in your favorite text editor and add your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
BRIGHT_DATA_MCP_KEY=your_bright_data_mcp_key
```

You'll need to get these API keys from Anthropic and Bright Data respectively. If you don't have them yet, you can sign up on their websites.

### Running Your First Agent (03:30 - 06:00)

Now that we have everything set up, let's run our first agent. The project provides several different agent architectures, but we'll start with the basic agent:

```bash
python main.py --mode basic
```

This will start the agent in interactive mode. You'll see a welcome message and a prompt where you can start chatting with the agent.

Let's try a simple query:

```
Hello! Can you tell me about yourself?
```

The agent will respond with information about itself and its capabilities. You can ask it questions, request information, or give it tasks to perform.

Let's try another query that uses one of the agent's tools:

```
What's the weather like in New York?
```

The agent will use its tools to fetch the current weather information for New York and present it to you.

### Basic Commands and Interactions (06:00 - 08:30)

The agent supports several special commands that you can use during a chat session. Let's try some of them:

To get help and see all available commands:

```
help
```

This will display a list of all available commands.

To view the current memory state:

```
memory
```

This shows what the agent remembers from your conversation.

To provide feedback on the agent's response:

```
feedback That was a helpful response, but I'd like more details next time.
```

This feedback helps the agent learn and improve over time.

To end the chat session:

```
exit
```

This will terminate the agent and return to the command line.

### Customizing the Agent (08:30 - 10:30)

Now, let's look at how you can customize the agent. The main script accepts several command-line arguments that allow you to configure the agent's behavior.

For example, to run the advanced agent with verbose logging:

```bash
python main.py --mode advanced --verbose
```

You can also specify the memory backend:

```bash
python main.py --mode distributed_memory --memory-backend redis
```

For more advanced customization, you can use the Python API. Let's create a simple script that configures and runs an agent:

```python
import asyncio
import os
from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent

# Set environment variables if needed
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
os.environ["BRIGHT_DATA_MCP_KEY"] = "your-mcp-key"

# Configure the agent
config = {
    "verbose": True,
    "memory_backend": "redis",
    "redis_url": "redis://localhost:6379/0",
    "model": "claude-3-sonnet",
    "max_tokens": 4096,
    "specialized_agents": ["research", "coding", "creative"],
    "context_window_size": 10
}

# Run the agent
asyncio.run(chat_with_advanced_enhanced_agent(config=config))
```

Save this as `custom_agent.py` and run it with:

```bash
python custom_agent.py
```

This gives you full control over the agent's configuration and behavior.

### Conclusion (10:30 - 11:30)

In this tutorial, we've covered how to install and set up DataMCPServerAgent, run your first agent, use basic commands and interactions, and customize the agent configuration.

You should now be able to start using DataMCPServerAgent for your own projects and experiments.

If you want to learn more, check out our other tutorials on creating custom tools, advanced memory management, and building multi-agent systems.

Thanks for watching, and happy coding!

## Visual Notes:

- [00:15] Show slide with DataMCPServerAgent features
- [01:30] Show terminal with git clone command
- [02:15] Show .env file being edited
- [04:00] Show agent starting up
- [05:00] Highlight the agent's response
- [06:30] Show list of commands from help
- [09:00] Show custom configuration code

## Resources:

- [DataMCPServerAgent GitHub Repository](https://github.com/DimaJoyti/DataMCPServerAgent)
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Bright Data MCP Documentation](https://brightdata.com/products/mcp)
- [DataMCPServerAgent Documentation](https://github.com/DimaJoyti/DataMCPServerAgent/tree/main/docs)