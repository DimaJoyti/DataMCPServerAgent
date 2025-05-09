# Enhanced Research Assistant

This document provides an overview of the Enhanced Research Assistant implementation for the DataMCPServerAgent project.

## Overview

The Enhanced Research Assistant is a sophisticated agent designed to help users gather, organize, and visualize information on various topics. It leverages advanced features such as memory persistence, tool selection, learning capabilities, and reinforcement learning integration to provide high-quality research results.

## Key Components

### 1. Memory Persistence

The Research Assistant uses a specialized memory persistence module (`ResearchMemoryDatabase`) to store and retrieve research data, including:

- Research projects
- Research queries
- Research results
- Sources
- Tool usage
- Visualizations

This allows the Research Assistant to maintain context across sessions and learn from past interactions.

### 2. Tool Selection

The Enhanced Tool Selector (`EnhancedToolSelector`) intelligently selects the most appropriate tools for a given research query based on:

- Query analysis
- Tool capabilities
- Past performance
- Learning feedback

The Tool Performance Tracker (`ToolPerformanceTracker`) monitors the performance of each tool, tracking metrics such as:

- Success rate
- Execution time
- Total usage

### 3. Learning Capabilities

The Research Assistant can learn from user feedback and self-evaluation to improve its performance over time. It collects and analyzes:

- User feedback on research results
- Tool performance metrics
- Query patterns
- Source quality

### 4. Reinforcement Learning Integration

The RL-Enhanced Research Assistant (`RLEnhancedResearchAssistant`) integrates reinforcement learning to continuously improve its tool selection and research quality. It uses:

- Q-learning for tool selection
- Reward system based on research quality
- State representation based on query analysis
- Learning from user feedback

## Usage

### Basic Usage

```python
from src.agents.enhanced_research_assistant import EnhancedResearchAssistant

# Initialize the research assistant
assistant = EnhancedResearchAssistant()

# Create a project
project = assistant.create_project(
    name="My Research Project",
    description="A project for researching various topics",
    tags=["research", "general"]
)

# Perform a research query
response = await assistant.invoke({
    "query": "What are the benefits of exercise?",
    "project_id": project.id,
    "citation_format": "apa"
})

# Parse the response
import json
output = json.loads(response["output"])

# Access the research results
topic = output["topic"]
summary = output["summary"]
sources = output["sources"]
tools_used = output["tools_used"]
bibliography = output.get("bibliography")
```

### Using the RL-Enhanced Research Assistant

```python
from src.agents.research_rl_integration import RLEnhancedResearchAssistant

# Initialize the RL-enhanced research assistant
assistant = RLEnhancedResearchAssistant(
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_rate=0.2
)

# Create a project
project = assistant.create_project(
    name="RL Research Project",
    description="A project for RL-enhanced research",
    tags=["research", "rl"]
)

# Perform a research query
response = await assistant.invoke({
    "query": "What are the latest advancements in artificial intelligence?",
    "project_id": project.id,
    "citation_format": "apa"
})

# Provide feedback
learning_results = await assistant.update_from_feedback(
    query="What are the latest advancements in artificial intelligence?",
    response=json.loads(response["output"]),
    feedback="This research was excellent and very comprehensive!"
)
```

### Command-Line Interface

The Research Assistant can be used from the command line using the provided main entry points:

```bash
# Run the Enhanced Research Assistant
python src/core/research_assistant_main.py

# Run the RL-Enhanced Research Assistant
python src/core/research_rl_main.py
```

## Features

- **Project Management**: Create and manage research projects
- **Query Handling**: Process research queries and store results
- **Source Management**: Track and cite sources properly
- **Tool Selection**: Intelligently select the most appropriate tools for a query
- **Memory Persistence**: Store and retrieve research data across sessions
- **Learning Capabilities**: Learn from user feedback and self-evaluation
- **Reinforcement Learning**: Continuously improve tool selection and research quality
- **Visualization**: Generate visualizations of research data
- **Export Options**: Export research results in various formats (Markdown, HTML, PDF, DOCX, PPTX)
- **Citation Formatting**: Format citations in various styles (APA, MLA, Chicago, Harvard, IEEE)

## Future Enhancements

- **Knowledge Graph Integration**: Integrate knowledge graphs for better context understanding
- **Distributed Memory**: Implement distributed memory for scalability
- **Multi-Agent Collaboration**: Enable collaboration between multiple research agents
- **Hierarchical Reinforcement Learning**: Implement hierarchical RL for complex research tasks
- **Automatic Option Discovery**: Automatically discover useful options based on agent's experience
- **Real-Time Learning**: Learn from user interactions in real-time
- **Advanced Visualization**: Enhance visualization capabilities with interactive charts and graphs
- **Natural Language Interface**: Improve the natural language interface for more intuitive interactions

## Testing

To test the Research Assistant, run the provided test script:

```bash
python src/tests/test_research_assistant.py
```

This will test both the Enhanced Research Assistant and the RL-Enhanced Research Assistant.
