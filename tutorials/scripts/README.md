# Tutorial Scripts for DataMCPServerAgent

This directory contains fully commented scripts that demonstrate various features of the DataMCPServerAgent. These scripts are designed to be run step-by-step with detailed explanations.

## Getting Started

To run the tutorial scripts, you'll need to have DataMCPServerAgent installed and configured. Make sure you have set up your environment variables in the `.env` file.

Then, you can run the scripts directly:

```bash
python 01_getting_started.py
```

## Available Scripts

### Basic Tutorials

1. **01_getting_started.py** - Introduction to the basic agent
   - Setting up and configuring the agent
   - Running the agent
   - Basic interactions

2. **02_advanced_agent.py** - Introduction to the advanced agent
   - Specialized agents
   - Context-aware memory
   - Advanced commands

3. **03_enhanced_agent.py** - Introduction to the enhanced agent
   - Learning capabilities
   - Feedback mechanisms
   - Performance metrics

### Custom Tools Tutorials

4. **04_basic_custom_tool.py** - Creating a basic custom tool
   - Creating a simple function-based tool
   - Registering the tool with the agent
   - Using the tool

5. **05_advanced_custom_tool.py** - Creating an advanced custom tool
   - Creating a class-based tool
   - Implementing synchronous and asynchronous methods
   - Error handling and validation

6. **06_tool_provider.py** - Creating a tool provider
   - Dynamic tool creation
   - Configuration-based tools
   - Tool lifecycle management

### Memory Management Tutorials

7. **07_memory_basics.py** - Basic memory management
   - Storing and retrieving memories
   - Memory persistence
   - Memory search

8. **08_distributed_memory.py** - Distributed memory
   - Redis backend
   - MongoDB backend
   - Memory caching

9. **09_knowledge_graph.py** - Knowledge graph integration
   - Entity and relationship modeling
   - Graph queries
   - Context enhancement

### Advanced Features Tutorials

10. **10_tool_selection.py** - Enhanced tool selection
    - Tool performance tracking
    - Adaptive tool selection
    - Execution feedback

11. **11_multi_agent_system.py** - Multi-agent learning system
    - Collaborative learning
    - Agent specialization
    - Knowledge sharing

12. **12_reinforcement_learning.py** - Reinforcement learning
    - Reward-based learning
    - Policy optimization
    - Continuous improvement

## Creating Your Own Tutorial Scripts

You can create your own tutorial scripts by following these guidelines:

1. Start with a clear objective and learning outcomes
2. Include detailed comments explaining each step
3. Use print statements to show progress and results
4. Handle errors gracefully
5. Include examples of both basic and advanced usage

Here's a template for a tutorial script:

```python
"""
Tutorial: [Title]

This tutorial demonstrates [brief description].

Learning objectives:
- [Objective 1]
- [Objective 2]
- [Objective 3]

Prerequisites:
- DataMCPServerAgent installed and configured
- [Other prerequisites]
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
from src.core.main import chat_with_agent

async def run_tutorial():
    """Run the tutorial."""
    print("Starting tutorial: [Title]")
    
    # Step 1: [Description]
    print("\nStep 1: [Description]")
    # Code for step 1
    
    # Step 2: [Description]
    print("\nStep 2: [Description]")
    # Code for step 2
    
    # Step 3: [Description]
    print("\nStep 3: [Description]")
    # Code for step 3
    
    print("\nTutorial completed!")

if __name__ == "__main__":
    asyncio.run(run_tutorial())
```

Feel free to use this template as a starting point for your own tutorial scripts.