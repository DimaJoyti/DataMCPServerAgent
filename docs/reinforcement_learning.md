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
- Persists rewards in the database

Implementation: `src/agents/reinforcement_learning.py`

### QLearningAgent

The `QLearningAgent` class implements the Q-learning algorithm for reinforcement learning. It learns to select actions (sub-agents) based on states (user requests) to maximize rewards.

Key features:

- Maintains a Q-table mapping states to action values
- Uses epsilon-greedy exploration strategy
- Updates Q-values based on rewards
- Persists Q-table in the database

Implementation: `src/agents/reinforcement_learning.py`

### PolicyGradientAgent

The `PolicyGradientAgent` class implements the policy gradient algorithm for reinforcement learning. It learns a policy that directly maps states to action probabilities.

Key features:

- Maintains policy parameters for each action
- Uses softmax to calculate action probabilities
- Updates policy parameters using gradient ascent
- Persists policy parameters in the database

Implementation: `src/agents/reinforcement_learning.py`

### RLCoordinatorAgent

The `RLCoordinatorAgent` class coordinates the reinforcement learning process. It uses either Q-learning or policy gradient to select sub-agents for handling user requests.

Key features:

- Extracts state representations from user requests
- Selects sub-agents using reinforcement learning
- Updates the RL agent based on rewards
- Supports batch learning from past interactions

Implementation: `src/agents/reinforcement_learning.py`

## Learning Process

The reinforcement learning process consists of the following steps:

1. **State Extraction**: The system extracts a state representation from the user request and conversation history.
2. **Action Selection**: The RL agent selects a sub-agent to handle the request based on the current policy.
3. **Execution**: The selected sub-agent processes the request.
4. **Reward Calculation**: The system calculates a reward based on the execution result and user feedback.
5. **Policy Update**: The RL agent updates its policy based on the reward.
6. **Persistence**: The updated policy is persisted in the database for future use.

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

See `examples/reinforcement_learning_example.py` for a complete example of using the reinforcement learning system:

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

# Create memory database
db = MemoryDatabase("reinforcement_learning_example.db")

# Create specialized sub-agents
sub_agents = await create_specialized_sub_agents(model, mcp_tools)

# Create RL coordinator agent
rl_coordinator = await create_rl_agent_architecture(
    model=model,
    db=db,
    sub_agents=sub_agents,
    rl_agent_type="q_learning"  # or "policy_gradient"
)

# Process a request
result = await rl_coordinator.process_request(
    "Search for information about reinforcement learning",
    history=[]
)

# Provide feedback
await rl_coordinator.update_from_feedback(
    request="Search for information about reinforcement learning",
    response=result["response"],
    feedback="That was very helpful, thank you!"
)

# Perform batch learning
learning_result = await rl_coordinator.learn_from_batch()
```

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

1. **Deep Q-Networks (DQN)**: Implement DQN for handling more complex state spaces.
2. **Proximal Policy Optimization (PPO)**: Implement PPO for more stable policy gradient learning.
3. **Multi-Agent Reinforcement Learning**: Extend to multiple agents learning collaboratively.
4. **Hierarchical Reinforcement Learning**: Implement hierarchical RL for handling complex tasks.
5. **Reward Shaping**: Improve reward calculation with more sophisticated metrics.
6. **State Representation Learning**: Implement better state representation extraction.
7. **Exploration Strategies**: Implement more sophisticated exploration strategies.
8. **Transfer Learning**: Enable knowledge transfer between different RL agents.
