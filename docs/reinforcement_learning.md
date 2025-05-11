# Reinforcement Learning

This document provides detailed information about the reinforcement learning capabilities in the DataMCPServerAgent project.

## Overview

The reinforcement learning module extends the agent architecture with continuous improvement capabilities. It enables agents to learn from rewards and improve their performance over time through experience.

## Components

The reinforcement learning system consists of the following components:

### RewardSystem

The `RewardSystem` class calculates rewards based on agent performance and user feedback. It uses a weighted combination of user satisfaction, task completion, and efficiency metrics to determine the reward value.

Key features:

- Calculates rewards based on multiple factors
- Tracks reward history
- Persists rewards in the database through the memory persistence system

Implementation: `src/agents/reinforcement_learning.py`

For detailed information about how rewards and other RL data are persisted, see [Reinforcement Learning Memory Persistence](reinforcement_learning_memory.md).

### QLearningAgent

The `QLearningAgent` class implements the Q-learning algorithm for reinforcement learning. It learns to select actions (sub-agents) based on states (user requests) to maximize rewards.

Key features:

- Maintains a Q-table mapping states to action values
- Uses epsilon-greedy exploration strategy
- Updates Q-values based on rewards
- Persists Q-table in the database through the memory persistence system
- Loads previously learned Q-tables when initialized

Implementation: `src/agents/reinforcement_learning.py`

### PolicyGradientAgent

The `PolicyGradientAgent` class implements the policy gradient algorithm for reinforcement learning. It learns a policy that directly maps states to action probabilities.

Key features:

- Maintains policy parameters for each action
- Uses softmax to calculate action probabilities
- Updates policy parameters using gradient ascent
- Persists policy parameters in the database through the memory persistence system
- Loads previously learned policy parameters when initialized
- Maintains episode history for batch updates

Implementation: `src/agents/reinforcement_learning.py`

### RLCoordinatorAgent

The `RLCoordinatorAgent` class coordinates the reinforcement learning process. It uses either Q-learning or policy gradient to select sub-agents for handling user requests.

Key features:

- Extracts state representations from user requests
- Selects sub-agents using reinforcement learning
- Updates the RL agent based on rewards
- Supports batch learning from past interactions stored in the memory database
- Persists agent decisions and interactions for analysis
- Coordinates memory persistence for all RL components

Implementation: `src/agents/reinforcement_learning.py`

## Learning Process

The reinforcement learning process consists of the following steps:

1. **Initialization**: The system loads previously learned policies and parameters from the memory database.
2. **State Extraction**: The system extracts a state representation from the user request and conversation history.
3. **Action Selection**: The RL agent selects a sub-agent to handle the request based on the current policy.
4. **Execution**: The selected sub-agent processes the request.
5. **Reward Calculation**: The system calculates a reward based on the execution result and user feedback.
6. **Policy Update**: The RL agent updates its policy based on the reward.
7. **Persistence**: The updated policy, rewards, and interaction data are persisted in the memory database for future use.
8. **Batch Learning**: Periodically, the system performs batch learning from past interactions stored in the memory database.

The memory persistence system ensures that learning is continuous across sessions and that the agent can improve over time based on accumulated experience. For detailed information about the memory persistence system, see [Reinforcement Learning Memory Persistence](reinforcement_learning_memory.md).

## Usage

To use the reinforcement learning agent, you can run the following command:

```bash
python -m src.core.reinforcement_learning_main
```

Or use the Python API:

```python
import asyncio
from src.core.reinforcement_learning_main import chat_with_rl_agent

asyncio.run(chat_with_rl_agent())
```

## Example

See `examples/reinforcement_learning_example.py` for a complete example of using the reinforcement learning system with memory persistence:

```python
import asyncio
from src.agents.reinforcement_learning import (
    RewardSystem,
    QLearningAgent,
    PolicyGradientAgent,
    RLCoordinatorAgent,
    create_rl_agent_architecture
)
from src.memory.memory_persistence import MemoryDatabase
from langchain_anthropic import ChatAnthropic

async def main():
    # Create memory database for persistence
    # This will create a SQLite database file if it doesn't exist
    # or load an existing one if it does
    db = MemoryDatabase("reinforcement_learning_example.db")

    # Create language model
    model = ChatAnthropic(model="claude-3-opus-20240229")

    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mcp_tools)

    # Create RL coordinator agent
    # This will load any previously learned policies from the database
    rl_coordinator = await create_rl_agent_architecture(
        model=model,
        db=db,
        sub_agents=sub_agents,
        rl_agent_type="q_learning"  # or "policy_gradient"
    )

    # Process a request
    # The state, action, and reward will be automatically persisted
    result = await rl_coordinator.process_request(
        "Search for information about reinforcement learning",
        history=[]
    )

    # Provide feedback
    # This will update the policy and persist the changes
    await rl_coordinator.update_from_feedback(
        request="Search for information about reinforcement learning",
        response=result["response"],
        feedback="That was very helpful, thank you!"
    )

    # Perform batch learning from past interactions stored in the database
    # This will load past interactions, update the policy, and persist the changes
    learning_result = await rl_coordinator.learn_from_batch(batch_size=10)

    print(f"Learning result: {learning_result}")

    # The next time the agent is started, it will load the learned policy
    # from the database and continue learning from where it left off

if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates how the reinforcement learning system uses memory persistence to:

1. Load previously learned policies from the database
2. Persist new learning as it occurs
3. Store interaction data for batch learning
4. Ensure continuous learning across sessions

## Configuration

The reinforcement learning system can be configured using the following environment variables:

- `RL_AGENT_TYPE`: Type of RL agent to use (`q_learning` or `policy_gradient`)
- `RL_LEARNING_RATE`: Learning rate for the RL agent
- `RL_DISCOUNT_FACTOR`: Discount factor for future rewards
- `RL_EXPLORATION_RATE`: Exploration rate for epsilon-greedy strategy

## Comparison of Approaches

### Q-Learning

Q-learning is a value-based reinforcement learning algorithm that learns the value of actions in different states. It is well-suited for discrete state spaces and simpler problems.

Advantages:

- Simple to implement and understand
- Works well with discrete state spaces
- Stable learning process
- Good for problems with clear state transitions

Disadvantages:

- Struggles with continuous state spaces
- May require state discretization
- Can be inefficient for large state spaces
- Limited by tabular representation

### Policy Gradient

Policy gradient is a policy-based reinforcement learning algorithm that directly learns a policy mapping states to actions. It is well-suited for continuous state spaces and more complex problems.

Advantages:

- Works well with continuous state spaces
- Can learn stochastic policies
- More efficient for large state spaces
- Can handle high-dimensional state spaces

Disadvantages:

- More complex to implement
- Can have high variance in learning
- May converge to local optima
- Requires careful hyperparameter tuning

## Future Improvements

Potential future improvements to the reinforcement learning system include:

### Algorithm Improvements

1. **Deep Q-Networks (DQN)**: Implement DQN for handling more complex state spaces.
2. **Proximal Policy Optimization (PPO)**: Implement PPO for more stable policy gradient learning.
3. **Multi-Agent Reinforcement Learning**: Extend to multiple agents learning collaboratively.
4. **Hierarchical Reinforcement Learning**: Implement hierarchical RL for handling complex tasks.
5. **Reward Shaping**: Improve reward calculation with more sophisticated metrics.
6. **State Representation Learning**: Implement better state representation extraction.
7. **Exploration Strategies**: Implement more sophisticated exploration strategies.
8. **Transfer Learning**: Enable knowledge transfer between different RL agents.

### Memory Persistence Improvements

1. **Distributed Memory**: Implement distributed memory storage for scalability.
2. **Knowledge Graph Integration**: Store RL data in a knowledge graph for better context understanding.
3. **Memory Compression**: Implement compression techniques for efficient storage of large Q-tables and policy parameters.
4. **Memory Versioning**: Add versioning for tracking changes to Q-tables and policies over time.
5. **Memory Pruning**: Implement pruning mechanisms for removing outdated or unused memory entries.
6. **Memory Encryption**: Add encryption for sensitive memory data.
7. **Memory Backup and Recovery**: Implement backup and recovery mechanisms for memory persistence.
8. **Memory Performance Optimization**: Optimize database queries and storage for better performance.

For more details on memory persistence improvements, see [Reinforcement Learning Memory Persistence](reinforcement_learning_memory.md).
