# Multi-Agent Learning System

This document provides detailed information about the multi-agent learning system in the DataMCPServerAgent project.

## Overview

The multi-agent learning system extends the agent architecture with collaborative learning and knowledge sharing between agents. It enables agents to learn from each other's experiences and collaborate effectively to solve complex problems.

## Components

The multi-agent learning system consists of the following components:

### MultiAgentLearningSystem

The `MultiAgentLearningSystem` class is the main entry point for the multi-agent learning system. It coordinates the learning and collaboration between agents.

Key features:
- Analyzes agent performance to identify opportunities for improvement
- Executes learning cycles to improve agent performance
- Processes requests collaboratively using multiple agents

Implementation: `src/agents/multi_agent_learning.py`

### CollaborativeLearningSystem

The `CollaborativeLearningSystem` class facilitates collaborative learning between agents. It enables agents to share knowledge and collaborate on problem-solving.

Key features:
- Develops collaboration strategies based on agent performance
- Shares knowledge between agents
- Enables collaborative problem-solving

Implementation: `src/agents/multi_agent_learning.py`

### KnowledgeTransferAgent

The `KnowledgeTransferAgent` class is responsible for transferring knowledge between agents. It extracts knowledge from agent interactions and adapts it for use by other agents.

Key features:
- Extracts valuable knowledge from agent interactions
- Adapts knowledge for use by different agent types
- Facilitates knowledge transfer between agents

Implementation: `src/agents/multi_agent_learning.py`

### CollaborativeKnowledgeBase

The `CollaborativeKnowledgeBase` class stores and manages knowledge shared between agents. It provides methods for storing, retrieving, and updating knowledge.

Key features:
- Stores knowledge items with metadata
- Tracks knowledge applicability to different agent types
- Records knowledge transfers between agents
- Assigns knowledge to agents with proficiency levels

Implementation: `src/memory/collaborative_knowledge.py`

### AgentPerformanceTracker

The `AgentPerformanceTracker` class tracks agent performance metrics. It provides methods for recording and analyzing agent performance.

Key features:
- Records agent execution metrics
- Calculates agent success rates and execution times
- Provides performance summaries for individual agents
- Tracks collaborative performance metrics

Implementation: `src/utils/agent_metrics.py`

### MultiAgentPerformanceAnalyzer

The `MultiAgentPerformanceAnalyzer` class analyzes agent performance to identify opportunities for improvement. It provides methods for analyzing agent synergy and identifying optimal agent combinations.

Key features:
- Analyzes synergy between agents
- Identifies optimal agent combinations
- Analyzes the impact of learning on agent performance

Implementation: `src/utils/agent_metrics.py`

## Knowledge Sharing Process

The multi-agent learning system enables knowledge sharing between agents through the following process:

1. **Knowledge Extraction**: The `KnowledgeTransferAgent` extracts valuable knowledge from agent interactions.
2. **Knowledge Storage**: The extracted knowledge is stored in the `CollaborativeKnowledgeBase`.
3. **Knowledge Adaptation**: The `KnowledgeTransferAgent` adapts the knowledge for use by different agent types.
4. **Knowledge Transfer**: The adapted knowledge is transferred to the target agent.
5. **Knowledge Application**: The target agent incorporates the knowledge to improve its performance.
6. **Performance Tracking**: The `AgentPerformanceTracker` tracks the impact of knowledge transfer on agent performance.

## Collaborative Problem-Solving

The multi-agent learning system enables collaborative problem-solving through the following process:

1. **Task Decomposition**: The `MultiAgentLearningCoordinator` breaks down complex tasks into subtasks.
2. **Agent Selection**: The coordinator selects the most appropriate agents for each subtask based on their capabilities and performance history.
3. **Subtask Execution**: Each selected agent executes its assigned subtask.
4. **Result Synthesis**: The coordinator synthesizes the results from all agents into a coherent response.
5. **Performance Tracking**: The `AgentPerformanceTracker` tracks the performance of collaborative problem-solving.

## Learning Cycle

The multi-agent learning system executes learning cycles to improve agent performance. A learning cycle consists of the following steps:

1. **Performance Analysis**: The `MultiAgentPerformanceAnalyzer` analyzes agent performance to identify opportunities for improvement.
2. **Collaboration Strategy Development**: The `CollaborativeLearningSystem` develops strategies for effective collaboration.
3. **Knowledge Transfer**: The `KnowledgeTransferAgent` transfers knowledge between agents based on the identified opportunities.
4. **Learning from Feedback**: Each agent learns from feedback collected during interactions.
5. **Performance Evaluation**: The system evaluates the impact of learning on agent performance.

## Usage

To use the multi-agent learning system, you can run the following command:

```bash
python main.py --mode multi_agent
```

Or use the Python API:

```python
import asyncio
from src.core.multi_agent_main import chat_with_multi_agent_learning_system

asyncio.run(chat_with_multi_agent_learning_system())
```

## Example

See `examples/multi_agent_learning_example.py` for a complete example of using the multi-agent learning system:

```python
import asyncio
from src.agents.learning_capabilities import FeedbackCollector, LearningAgent
from src.agents.multi_agent_learning import (
    CollaborativeLearningSystem,
    KnowledgeTransferAgent,
    MultiAgentLearningSystem
)
from src.memory.collaborative_knowledge import CollaborativeKnowledgeBase
from src.memory.memory_persistence import MemoryDatabase
from src.utils.agent_metrics import AgentPerformanceTracker, MultiAgentPerformanceAnalyzer

# Create memory database
db = MemoryDatabase("example_multi_agent_memory.db")

# Create collaborative knowledge base
knowledge_base = CollaborativeKnowledgeBase(db)

# Create performance tracker
performance_tracker = AgentPerformanceTracker(db)

# Create learning agents
learning_agents = {
    "search_agent": LearningAgent("search_agent", model, db, feedback_collector),
    "scraping_agent": LearningAgent("scraping_agent", model, db, feedback_collector),
    "analysis_agent": LearningAgent("analysis_agent", model, db, feedback_collector)
}

# Create multi-agent learning system
multi_agent_learning = MultiAgentLearningSystem(
    model, db, learning_agents, feedback_collector
)

# Execute a learning cycle
learning_results = await multi_agent_learning.execute_learning_cycle()

# Process a request collaboratively
collaborative_solution = await multi_agent_learning.process_request_collaboratively(
    "Research and analyze multi-agent learning systems",
    agent_results
)
```