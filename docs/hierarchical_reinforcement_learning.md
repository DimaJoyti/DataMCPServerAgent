# Hierarchical Reinforcement Learning

This document provides detailed information about the hierarchical reinforcement learning capabilities in the DataMCPServerAgent project.

## Overview

Hierarchical Reinforcement Learning (HRL) extends traditional reinforcement learning by introducing temporal abstraction and hierarchical control structures. This enables the agent to handle complex, multi-step tasks by breaking them down into simpler subtasks and learning at multiple levels of abstraction.

## Key Concepts

### Temporal Abstraction

Temporal abstraction allows the agent to reason and learn at multiple time scales. Instead of making decisions at each primitive time step, the agent can make decisions at various levels of temporal abstraction.

### Options Framework

The options framework is a formalism for temporal abstraction in reinforcement learning. An option consists of:

1. **Initiation Set**: States where the option can be initiated
2. **Policy**: Mapping from states to actions within the option
3. **Termination Condition**: Conditions for terminating the option

### Hierarchical Structure

The hierarchical structure consists of multiple levels:

1. **Top Level**: Selects options (temporally extended actions)
2. **Middle Levels**: May select sub-options or primitive actions
3. **Bottom Level**: Executes primitive actions

## Components

### Hierarchical Memory Database

The `HierarchicalMemoryDatabase` class (`src/memory/hierarchical_memory_persistence.py`) extends the `AdvancedMemoryDatabase` with support for hierarchical reinforcement learning:

- Stores options and their policies
- Tracks subtask execution history
- Maintains hierarchical Q-tables
- Records task decomposition

### Hierarchical Reward System

The `HierarchicalRewardSystem` class (`src/agents/hierarchical_rl.py`) extends the basic `RewardSystem` with hierarchical reward calculation:

- Calculates rewards at different levels of the hierarchy
- Adjusts rewards based on the level (higher levels emphasize task completion, lower levels emphasize efficiency)
- Tracks reward history for each level

### Option Class

The `Option` class (`src/agents/hierarchical_rl.py`) represents a temporally extended action:

- Defines initiation conditions
- Implements a policy for action selection
- Specifies termination conditions
- Provides methods for option execution

### Hierarchical Q-Learning Agent

The `HierarchicalQLearningAgent` class (`src/agents/hierarchical_rl.py`) implements hierarchical Q-learning:

- Maintains Q-tables at multiple levels
- Selects options at the top level
- Selects primitive actions at the bottom level
- Updates Q-values at all levels

### Hierarchical RL Coordinator Agent

The `HierarchicalRLCoordinatorAgent` class (`src/agents/hierarchical_rl.py`) coordinates the hierarchical reinforcement learning process:

- Decomposes tasks into subtasks
- Selects options using hierarchical RL
- Executes options until termination
- Tracks performance at multiple levels

## Task Decomposition

Task decomposition is a key aspect of hierarchical reinforcement learning. The agent breaks down complex tasks into simpler subtasks, which can be handled more effectively.

### Task Decomposition Process

1. **Task Analysis**: The agent analyzes the user request to identify the main task
2. **Subtask Identification**: The agent identifies subtasks that need to be completed
3. **Dependency Specification**: The agent determines dependencies between subtasks
4. **Execution Planning**: The agent plans the execution of subtasks based on dependencies

### Task Decomposition Prompt

The task decomposition is performed using a language model with a specialized prompt:

```python
task_decomposition_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a task decomposition agent responsible for breaking down complex tasks into simpler subtasks.
Your job is to analyze a user request and decompose it into a sequence of subtasks that can be executed to fulfill the request.

For each request, you should:
1. Identify the main task
2. Break it down into subtasks
3. Specify the dependencies between subtasks
4. Provide a brief description for each subtask

Respond with a JSON object containing the task decomposition.
"""
        ),
        HumanMessage(
            content="""
User request:
{request}

Recent conversation:
{history}

Decompose this task into subtasks.
"""
        ),
    ]
)
```

## Option Discovery and Learning

Options can be predefined or discovered automatically through option discovery algorithms.

### Predefined Options

Predefined options are created based on domain knowledge:

```python
option = Option.create_option(
    option_name="Use search_agent",
    initiation_states=["search_query", "information_request"],
    termination_states=["task_completed", "error_state"],
    policy_mapping={"search_query": "search_agent", "information_request": "search_agent"},
    db=db,
    agent_name="hierarchical_rl_demo",
)
```

### Automatic Option Discovery

Automatic option discovery can be enabled using the `RL_OPTION_DISCOVERY_ENABLED` environment variable. When enabled, the agent will discover options based on:

1. **Bottleneck States**: States that are visited frequently in successful episodes
2. **Subgoal Discovery**: Identifying subgoals that can be reached reliably
3. **Skill Chaining**: Learning sequences of skills that can be composed

## Configuration

The hierarchical reinforcement learning system can be configured using the following environment variables:

- `RL_MODE=hierarchical`: Set the RL mode to hierarchical
- `RL_HIERARCHY_LEVELS=2`: Number of hierarchy levels
- `RL_TASK_DECOMPOSITION_ENABLED=true`: Whether to enable task decomposition
- `RL_OPTION_DISCOVERY_ENABLED=false`: Whether to enable automatic option discovery

## Usage

### Basic Usage

```python
from src.core.reinforcement_learning_main import chat_with_rl_agent
import os

# Set environment variables
os.environ["RL_MODE"] = "hierarchical"
os.environ["RL_HIERARCHY_LEVELS"] = "2"
os.environ["RL_TASK_DECOMPOSITION_ENABLED"] = "true"

# Start the hierarchical RL agent
asyncio.run(chat_with_rl_agent())
```

### Advanced Usage

```python
import asyncio
from src.agents.hierarchical_rl import create_hierarchical_rl_agent_architecture
from src.memory.hierarchical_memory_persistence import HierarchicalMemoryDatabase

# Initialize hierarchical memory database
db = HierarchicalMemoryDatabase("agent_memory.db")

# Create hierarchical RL coordinator agent
hierarchical_rl_coordinator = await create_hierarchical_rl_agent_architecture(
    model=model,
    db=db,
    sub_agents=sub_agents,
    tools=tools,
)

# Process a request
result = await hierarchical_rl_coordinator.process_request(
    "I need to plan a trip to Paris. Find the weather, translate some basic phrases, and summarize the top attractions.",
    history=[]
)
```

## Example

See `examples/hierarchical_rl_example.py` for a complete example of using the hierarchical reinforcement learning capabilities.

## Advantages of Hierarchical RL

Hierarchical reinforcement learning offers several advantages over flat RL approaches:

1. **Improved Sample Efficiency**: Learning at multiple levels of abstraction reduces the number of samples needed
2. **Better Exploration**: Temporal abstraction enables more efficient exploration of the state space
3. **Transfer Learning**: Skills learned for one task can be transferred to other tasks
4. **Handling Complex Tasks**: Breaking down complex tasks into simpler subtasks makes them more manageable
5. **Interpretability**: The hierarchical structure provides better interpretability of the agent's decisions

## Limitations and Challenges

Hierarchical reinforcement learning also faces some challenges:

1. **Option Design**: Designing effective options can be challenging and may require domain knowledge
2. **Credit Assignment**: Assigning credit for success or failure across multiple levels is complex
3. **Increased Complexity**: The hierarchical structure adds complexity to the learning algorithm
4. **Suboptimal Solutions**: The hierarchical structure may lead to suboptimal solutions in some cases

## Future Directions

Potential future enhancements to the hierarchical reinforcement learning system:

1. **Improved Option Discovery**: Develop more sophisticated algorithms for automatic option discovery
2. **Meta-Learning for Option Creation**: Use meta-learning to create options based on task requirements
3. **Hierarchical Imitation Learning**: Combine hierarchical RL with imitation learning for faster skill acquisition
4. **Multi-Agent Hierarchical RL**: Extend to multi-agent settings with hierarchical coordination
5. **Explainable Hierarchical RL**: Enhance explainability of hierarchical decision-making

## References

1. Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112(1-2), 181-211.
2. Bacon, P. L., Harb, J., & Precup, D. (2017). The option-critic architecture. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 31, No. 1).
3. Kulkarni, T. D., Narasimhan, K., Saeedi, A., & Tenenbaum, J. B. (2016). Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. Advances in Neural Information Processing Systems, 29.
4. Nachum, O., Gu, S. S., Lee, H., & Levine, S. (2018). Data-efficient hierarchical reinforcement learning. Advances in Neural Information Processing Systems, 31.
