"""
Hierarchical reinforcement learning module for DataMCPServerAgent.
This module implements hierarchical reinforcement learning for handling complex, multi-step tasks.
"""

import random
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from src.agents.advanced_rl_decision_making import AdvancedRLCoordinatorAgent
from src.agents.reinforcement_learning import RewardSystem
from src.memory.hierarchical_memory_persistence import HierarchicalMemoryDatabase

class HierarchicalRewardSystem(RewardSystem):
    """System for calculating rewards in a hierarchical reinforcement learning setting."""

    def __init__(self, db: HierarchicalMemoryDatabase):
        """Initialize the hierarchical reward system.

        Args:
            db: Hierarchical memory database for persistence
        """
        super().__init__(db)
        self.db = db
        self.reward_history = {}

    def calculate_hierarchical_reward(
        self,
        agent_name: str,
        task_id: str,
        parent_task_id: Optional[str],
        subtask_name: str,
        feedback: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        level: int,
    ) -> float:
        """Calculate reward for a subtask in a hierarchical setting.

        Args:
            agent_name: Name of the agent
            task_id: Unique identifier for the task
            parent_task_id: Identifier for the parent task (None for top-level tasks)
            subtask_name: Name of the subtask
            feedback: User feedback and self-evaluation
            performance_metrics: Performance metrics
            level: Hierarchy level (0 for top level)

        Returns:
            Calculated reward
        """
        # Calculate base reward using the parent class method
        base_reward = self.calculate_reward(agent_name, feedback, performance_metrics)

        # Apply level-specific adjustments
        # Higher levels get more weight for task completion and user satisfaction
        # Lower levels get more weight for efficiency and accuracy
        if level == 0:  # Top level
            # Emphasize task completion and user satisfaction
            adjusted_reward = (
                base_reward * 0.6
                + self._calculate_task_completion(performance_metrics) * 0.3
                + self._calculate_user_satisfaction(feedback) * 0.1
            )
        else:  # Lower levels
            # Emphasize efficiency and accuracy
            adjusted_reward = (
                base_reward * 0.6
                + self._calculate_efficiency(performance_metrics) * 0.2
                + self._calculate_accuracy(feedback, performance_metrics) * 0.2
            )

        # Store the reward in history
        if agent_name not in self.reward_history:
            self.reward_history[agent_name] = []

        self.reward_history[agent_name].append(
            {
                "timestamp": time.time(),
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                "subtask_name": subtask_name,
                "reward": adjusted_reward,
                "level": level,
            }
        )

        return adjusted_reward

class Option:
    """Represents a temporally extended action (option) in hierarchical reinforcement learning."""

    def __init__(
        self,
        option_id: str,
        option_name: str,
        initiation_set: Callable[[str], bool],
        termination_condition: Callable[[str, Dict[str, Any]], bool],
        policy: Callable[[str], str],
        db: HierarchicalMemoryDatabase,
        agent_name: str,
    ):
        """Initialize the option.

        Args:
            option_id: Unique identifier for the option
            option_name: Human-readable name for the option
            initiation_set: Function that determines if the option can be initiated in a state
            termination_condition: Function that determines if the option should terminate
            policy: Function that selects actions within the option
            db: Hierarchical memory database for persistence
            agent_name: Name of the agent
        """
        self.option_id = option_id
        self.option_name = option_name
        self.initiation_set = initiation_set
        self.termination_condition = termination_condition
        self.policy = policy
        self.db = db
        self.agent_name = agent_name

    def can_initiate(self, state: str) -> bool:
        """Check if the option can be initiated in the given state.

        Args:
            state: Current state

        Returns:
            True if the option can be initiated, False otherwise
        """
        return self.initiation_set(state)

    def should_terminate(self, state: str, result: Dict[str, Any]) -> bool:
        """Check if the option should terminate.

        Args:
            state: Current state
            result: Result of the last action

        Returns:
            True if the option should terminate, False otherwise
        """
        return self.termination_condition(state, result)

    def select_action(self, state: str) -> str:
        """Select an action according to the option's policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        return self.policy(state)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the option to a dictionary for storage.

        Returns:
            Dictionary representation of the option
        """
        return {
            "option_id": self.option_id,
            "option_name": self.option_name,
            # We can't directly serialize the functions, so we'll store their descriptions
            "initiation_set": {"description": "Initiation set function"},
            "termination_condition": {"description": "Termination condition function"},
            "policy": {"description": "Policy function"},
        }

    @classmethod
    def create_option(
        cls,
        option_name: str,
        initiation_states: List[str],
        termination_states: List[str],
        policy_mapping: Dict[str, str],
        db: HierarchicalMemoryDatabase,
        agent_name: str,
    ) -> "Option":
        """Create an option with simple state-based conditions.

        Args:
            option_name: Human-readable name for the option
            initiation_states: List of states where the option can be initiated
            termination_states: List of states where the option terminates
            policy_mapping: Mapping from states to actions
            db: Hierarchical memory database for persistence
            agent_name: Name of the agent

        Returns:
            Created option
        """
        option_id = str(uuid.uuid4())

        # Create simple functions based on the provided lists and mappings
        def initiation_set(state: str) -> bool:
            return state in initiation_states

        def termination_condition(state: str, result: Dict[str, Any]) -> bool:
            return state in termination_states or not result.get("success", False)

        def policy(state: str) -> str:
            return policy_mapping.get(state, random.choice(list(policy_mapping.values())))

        # Create and save the option
        option = cls(
            option_id=option_id,
            option_name=option_name,
            initiation_set=initiation_set,
            termination_condition=termination_condition,
            policy=policy,
            db=db,
            agent_name=agent_name,
        )

        # Save the option to the database
        db.save_option(
            agent_name=agent_name,
            option_id=option_id,
            option_name=option_name,
            initiation_set={"states": initiation_states},
            termination_condition={"states": termination_states},
            policy={"mapping": policy_mapping},
        )

        return option

class HierarchicalQLearningAgent:
    """Agent that learns using hierarchical Q-learning algorithm."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: HierarchicalMemoryDatabase,
        reward_system: HierarchicalRewardSystem,
        state_extractor: Callable[[Dict[str, Any]], str],
        primitive_actions: List[str],
        options: List[Option],
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,
        levels: int = 2,
    ):
        """Initialize the hierarchical Q-learning agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Hierarchical memory database for persistence
            reward_system: Hierarchical reward system
            state_extractor: Function to extract state from context
            primitive_actions: List of primitive actions
            options: List of options (temporally extended actions)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_rate: Exploration rate (epsilon)
            levels: Number of hierarchy levels
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.state_extractor = state_extractor
        self.primitive_actions = primitive_actions
        self.options = {option.option_id: option for option in options}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.levels = levels

        # Initialize hierarchical Q-tables
        self.q_tables = {}
        for level in range(levels):
            self.q_tables[level] = self.db.get_hierarchical_q_table(name, level) or {}

        # At the top level, actions are options
        self.top_level_actions = list(self.options.keys())

        # At the bottom level, actions are primitive actions
        self.bottom_level_actions = primitive_actions

    def select_option(self, state: str) -> str:
        """Select an option using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected option ID
        """
        # Exploration: random option
        if random.random() < self.exploration_rate:
            # Filter options that can be initiated in this state
            valid_options = [
                option_id
                for option_id, option in self.options.items()
                if option.can_initiate(state)
            ]
            if not valid_options:
                # If no valid options, return a random one
                return random.choice(self.top_level_actions)
            return random.choice(valid_options)

        # Exploitation: best option from Q-table
        return self._get_best_action(state, level=0)

    def select_primitive_action(self, state: str) -> str:
        """Select a primitive action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected primitive action
        """
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.choice(self.bottom_level_actions)

        # Exploitation: best action from Q-table
        return self._get_best_action(state, level=self.levels - 1)

    def _get_best_action(self, state: str, level: int) -> str:
        """Get the best action for a state from the Q-table at the specified level.

        Args:
            state: Current state
            level: Hierarchy level

        Returns:
            Best action
        """
        # If state not in Q-table, initialize it
        if state not in self.q_tables[level]:
            actions = (
                self.top_level_actions if level == 0 else self.bottom_level_actions
            )
            self.q_tables[level][state] = {action: 0.0 for action in actions}

        # Get action with highest Q-value
        state_actions = self.q_tables[level][state]
        return max(state_actions, key=state_actions.get)

    def update_q_value(
        self, state: str, action: str, reward: float, next_state: str, level: int
    ) -> None:
        """Update Q-value using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            level: Hierarchy level
        """
        # If state not in Q-table, initialize it
        if state not in self.q_tables[level]:
            actions = (
                self.top_level_actions if level == 0 else self.bottom_level_actions
            )
            self.q_tables[level][state] = {action: 0.0 for action in actions}

        # If next_state not in Q-table, initialize it
        if next_state not in self.q_tables[level]:
            actions = (
                self.top_level_actions if level == 0 else self.bottom_level_actions
            )
            self.q_tables[level][next_state] = {action: 0.0 for action in actions}

        # Get current Q-value
        current_q = self.q_tables[level][state].get(action, 0.0)

        # Get max Q-value for next state
        max_next_q = max(self.q_tables[level][next_state].values())

        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Update Q-table
        self.q_tables[level][state][action] = new_q

        # Save Q-table to database
        self.db.save_hierarchical_q_table(self.name, level, self.q_tables[level])

    async def execute_option(
        self,
        option_id: str,
        initial_state: str,
        context: Dict[str, Any],
        task_id: str,
        parent_task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute an option until termination.

        Args:
            option_id: ID of the option to execute
            initial_state: Initial state
            context: Context dictionary
            task_id: Unique identifier for the task
            parent_task_id: Identifier for the parent task

        Returns:
            Result of the option execution
        """
        option = self.options.get(option_id)
        if not option:
            return {
                "success": False,
                "error": f"Option {option_id} not found",
            }

        # Check if the option can be initiated in this state
        if not option.can_initiate(initial_state):
            return {
                "success": False,
                "error": f"Option {option.option_name} cannot be initiated in state {initial_state}",
            }

        # Execute the option
        state = initial_state
        cumulative_reward = 0.0
        start_time = time.time()
        steps = []

        while True:
            # Select action according to the option's policy
            action = option.select_action(state)

            # Execute the action
            step_start_time = time.time()
            result = await self._execute_primitive_action(action, context)
            step_end_time = time.time()

            # Calculate reward
            reward = self.reward_system.calculate_hierarchical_reward(
                agent_name=self.name,
                task_id=task_id,
                parent_task_id=parent_task_id,
                subtask_name=option.option_name,
                feedback={"self_evaluation": result.get("self_evaluation", {})},
                performance_metrics={
                    "success_rate": 1.0 if result.get("success", False) else 0.0,
                    "response_time": step_end_time - step_start_time,
                },
                level=1,  # Options are at level 1
            )

            # Update Q-value for the primitive action
            next_state = await self.state_extractor(context)
            self.update_q_value(state, action, reward, next_state, level=1)

            # Save subtask execution
            self.db.save_subtask_execution(
                agent_name=self.name,
                task_id=task_id,
                parent_task_id=parent_task_id,
                subtask_name=f"{option.option_name}_{action}",
                state=state,
                action=action,
                reward=reward,
                success=result.get("success", False),
                start_time=step_start_time,
                end_time=step_end_time,
                metadata={"option_id": option_id},
            )

            # Add step to the list
            steps.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "success": result.get("success", False),
                }
            )

            # Update state and cumulative reward
            state = next_state
            cumulative_reward += reward

            # Check if the option should terminate
            if option.should_terminate(state, result):
                break

        # Calculate total duration
        end_time = time.time()
        duration = end_time - start_time

        # Return the result
        return {
            "success": True,
            "option_id": option_id,
            "option_name": option.option_name,
            "steps": steps,
            "cumulative_reward": cumulative_reward,
            "duration": duration,
            "final_state": state,
        }

    async def _execute_primitive_action(
        self, action: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a primitive action.

        Args:
            action: Primitive action to execute
            context: Context dictionary

        Returns:
            Result of the action execution
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would execute the primitive action
        # For now, we'll just return a mock result
        return {
            "success": True,
            "action": action,
            "response": f"Executed action {action}",
            "self_evaluation": {"accuracy": 0.8},
        }

class HierarchicalRLCoordinatorAgent:
    """Coordinator agent that uses hierarchical reinforcement learning for decision making."""

    def __init__(
        self,
        name: str,
        model: ChatAnthropic,
        db: HierarchicalMemoryDatabase,
        reward_system: HierarchicalRewardSystem,
        sub_agents: Dict[str, Any],
        tools: List[BaseTool],
        task_decomposition_prompt: ChatPromptTemplate,
    ):
        """Initialize the hierarchical RL coordinator agent.

        Args:
            name: Name of the agent
            model: Language model to use
            db: Hierarchical memory database for persistence
            reward_system: Hierarchical reward system
            sub_agents: Dictionary of sub-agents
            tools: List of available tools
            task_decomposition_prompt: Prompt for task decomposition
        """
        self.name = name
        self.model = model
        self.db = db
        self.reward_system = reward_system
        self.sub_agents = sub_agents
        self.tools = tools
        self.task_decomposition_prompt = task_decomposition_prompt

        # Create options for each sub-agent
        self.options = self._create_options()

        # Create hierarchical Q-learning agent
        self.rl_agent = HierarchicalQLearningAgent(
            name=f"{name}_hql",
            model=model,
            db=db,
            reward_system=reward_system,
            state_extractor=self._extract_state,
            primitive_actions=list(sub_agents.keys()),
            options=list(self.options.values()),
            levels=2,  # 2-level hierarchy: options and primitive actions
        )

        # Create the prompt for state extraction
        self.state_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a state extraction agent responsible for converting user requests into state representations.
Your job is to analyze a user request and extract key features that can be used to determine the appropriate agent to handle it.

For each request, you should:
1. Identify the main task type (search, scraping, analysis, etc.)
2. Recognize entities mentioned in the request
3. Determine the complexity level
4. Identify any special requirements

Respond with a concise state identifier that captures these key aspects.
"""
                ),
                HumanMessage(
                    content="""
User request:
{request}

Recent conversation:
{history}

Extract a state identifier for this request.
"""
                ),
            ]
        )

    def _create_options(self) -> Dict[str, Option]:
        """Create options for each sub-agent.

        Returns:
            Dictionary of options
        """
        options = {}
        for agent_name, agent in self.sub_agents.items():
            # Create an option for each sub-agent
            option = Option.create_option(
                option_name=f"Use {agent_name}",
                initiation_states=[],  # Will be populated later
                termination_states=[],  # Will be populated later
                policy_mapping={},  # Will be populated later
                db=self.db,
                agent_name=self.name,
            )
            options[option.option_id] = option
        return options

    async def _extract_state(self, context: Dict[str, Any]) -> str:
        """Extract state from context.

        Args:
            context: Context dictionary

        Returns:
            State identifier
        """
        # Extract request and history from context
        request = context.get("request", "")
        history = context.get("history", [])

        # Format history
        formatted_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-3:]]
        )

        # Prepare the input for the state extraction prompt
        input_values = {"request": request, "history": formatted_history}

        # Get the state identifier from the model
        messages = self.state_extraction_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Return the state identifier
        return response.content.strip()

    async def _decompose_task(
        self, request: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Decompose a task into subtasks.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Task decomposition result
        """
        # Format history
        formatted_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in history[-3:]]
        )

        # Prepare the input for the task decomposition prompt
        input_values = {"request": request, "history": formatted_history}

        # Get the task decomposition from the model
        messages = self.task_decomposition_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Parse the response
        # In a real implementation, this would parse the model's response
        # For now, we'll just return a mock result
        task_id = str(uuid.uuid4())
        subtasks = [
            {"name": "Subtask 1", "description": "First subtask"},
            {"name": "Subtask 2", "description": "Second subtask"},
            {"name": "Subtask 3", "description": "Third subtask"},
        ]

        # Save the task decomposition
        self.db.save_task_decomposition(
            agent_name=self.name,
            task_id=task_id,
            parent_task_id=None,
            task_name=request,
            subtasks=subtasks,
        )

        return {
            "task_id": task_id,
            "task_name": request,
            "subtasks": subtasks,
        }

    async def process_request(
        self, request: str, history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process a user request using hierarchical reinforcement learning.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Processing result
        """
        # Create context for state extraction
        context = {"request": request, "history": history}

        # Extract state
        state = await self._extract_state(context)

        # Decompose the task
        task_decomposition = await self._decompose_task(request, history)
        task_id = task_decomposition["task_id"]

        # Select option using hierarchical RL
        option_id = self.rl_agent.select_option(state)

        # Execute the option
        start_time = time.time()
        option_result = await self.rl_agent.execute_option(
            option_id=option_id,
            initial_state=state,
            context=context,
            task_id=task_id,
        )
        end_time = time.time()

        # Calculate reward for the option
        reward = self.reward_system.calculate_hierarchical_reward(
            agent_name=self.name,
            task_id=task_id,
            parent_task_id=None,
            subtask_name=option_result.get("option_name", ""),
            feedback={"self_evaluation": {}},
            performance_metrics={
                "success_rate": 1.0 if option_result.get("success", False) else 0.0,
                "response_time": end_time - start_time,
            },
            level=0,  # Top level
        )

        # Update Q-value for the option
        next_state = await self._extract_state(context)
        self.rl_agent.update_q_value(state, option_id, reward, next_state, level=0)

        # Return the result
        return {
            "success": option_result.get("success", False),
            "response": "Processed request using hierarchical RL",
            "selected_option": option_result.get("option_name", ""),
            "reward": reward,
            "task_id": task_id,
            "subtasks": task_decomposition["subtasks"],
        }

# Factory function to create hierarchical RL-based agent architecture
async def create_hierarchical_rl_agent_architecture(
    model: ChatAnthropic,
    db: HierarchicalMemoryDatabase,
    sub_agents: Dict[str, Any],
    tools: List[BaseTool],
) -> HierarchicalRLCoordinatorAgent:
    """Create a hierarchical reinforcement learning-based agent architecture.

    Args:
        model: Language model to use
        db: Hierarchical memory database for persistence
        sub_agents: Dictionary of sub-agents
        tools: List of available tools

    Returns:
        Hierarchical RL coordinator agent
    """
    # Create task decomposition prompt
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

    # Create hierarchical reward system
    reward_system = HierarchicalRewardSystem(db)

    # Create hierarchical RL coordinator agent
    hierarchical_rl_coordinator = HierarchicalRLCoordinatorAgent(
        name="hierarchical_rl_coordinator",
        model=model,
        db=db,
        reward_system=reward_system,
        sub_agents=sub_agents,
        tools=tools,
        task_decomposition_prompt=task_decomposition_prompt,
    )

    return hierarchical_rl_coordinator
