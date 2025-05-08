"""
Example script demonstrating the advanced reinforcement learning decision-making capabilities.
"""

import asyncio
import os
import sys
import time
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from src.agents.advanced_rl_decision_making import (
    AdvancedRLCoordinatorAgent,
    DeepRLAgent,
    create_advanced_rl_agent_architecture
)
from src.agents.agent_architecture import (
    SpecializedSubAgent,
    create_specialized_sub_agents
)
from src.agents.multi_objective_rl import (
    MultiObjectiveRLCoordinatorAgent,
    create_multi_objective_rl_agent_architecture
)
from src.agents.reinforcement_learning import (
    RewardSystem,
    RLCoordinatorAgent,
    create_rl_agent_architecture
)
from src.memory.memory_persistence import MemoryDatabase
from src.utils.decision_explanation import (
    DecisionExplainer,
    PolicyExplainer,
    QValueVisualizer
)
from src.utils.error_handlers import format_error_for_user
from src.utils.rl_ab_testing import RLABTestingFramework

load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize memory database
db = MemoryDatabase("advanced_rl_example.db")

# Initialize decision explainer
decision_explainer = DecisionExplainer(
    model=model,
    db=db,
    explanation_level="detailed"
)

# Initialize policy explainer
policy_explainer = PolicyExplainer(model=model, db=db)

# Initialize Q-value visualizer
q_value_visualizer = QValueVisualizer(db=db)


async def create_mock_tools() -> List[BaseTool]:
    """Create mock tools for testing.
    
    Returns:
        List of mock tools
    """
    # Create mock tools
    return [
        BaseTool(
            name="search_tool",
            description="Search for information on the web",
            func=lambda x: "Search results for: " + x
        ),
        BaseTool(
            name="calculator_tool",
            description="Perform calculations",
            func=lambda x: "Calculation result: " + str(eval(x))
        ),
        BaseTool(
            name="weather_tool",
            description="Get weather information",
            func=lambda x: "Weather for " + x + ": Sunny, 75°F"
        ),
        BaseTool(
            name="translation_tool",
            description="Translate text between languages",
            func=lambda x: "Translation: " + x
        ),
        BaseTool(
            name="summarization_tool",
            description="Summarize long text",
            func=lambda x: "Summary: " + x[:50] + "..."
        ),
    ]


async def demonstrate_basic_rl() -> None:
    """Demonstrate basic reinforcement learning."""
    print("\n=== Demonstrating Basic Reinforcement Learning ===\n")
    
    # Create mock tools
    mock_tools = await create_mock_tools()
    
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mock_tools)
    
    # Create basic RL coordinator agent
    rl_coordinator = await create_rl_agent_architecture(
        model=model,
        db=db,
        sub_agents=sub_agents,
        rl_agent_type="q_learning"
    )
    
    # Process a few requests
    requests = [
        "What is the weather in New York?",
        "Calculate 15 * 7",
        "Translate 'hello' to Spanish",
        "Search for information about reinforcement learning",
        "Summarize this article about climate change",
    ]
    
    for request in requests:
        print(f"\nProcessing request: {request}")
        result = await rl_coordinator.process_request(request, [])
        print(f"Selected agent: {result['selected_agent']}")
        print(f"Reward: {result['reward']:.4f}")
    
    # Get Q-table summary
    state = "weather_query"  # Example state
    q_value_summary = q_value_visualizer.get_q_value_summary(
        "rl_coordinator_q_learning", state, list(sub_agents.keys())
    )
    
    print("\nQ-Value Summary:")
    print(f"State: {q_value_summary['state']}")
    print(f"Q-values: {q_value_summary['q_values']}")
    print(f"Best action: {q_value_summary['max_action']}")


async def demonstrate_advanced_rl() -> None:
    """Demonstrate advanced reinforcement learning with deep RL."""
    print("\n=== Demonstrating Advanced Reinforcement Learning ===\n")
    
    # Create mock tools
    mock_tools = await create_mock_tools()
    
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mock_tools)
    
    # Create advanced RL coordinator agent
    advanced_rl_coordinator = await create_advanced_rl_agent_architecture(
        model=model,
        db=db,
        sub_agents=sub_agents,
        tools=mock_tools,
        rl_agent_type="deep_rl"
    )
    
    # Process a few requests
    requests = [
        "What is the weather in Los Angeles?",
        "Calculate the square root of 144",
        "Translate 'goodbye' to French",
        "Search for information about deep reinforcement learning",
        "Summarize this research paper on neural networks",
    ]
    
    for request in requests:
        print(f"\nProcessing request: {request}")
        result = await advanced_rl_coordinator.process_request(request, [])
        print(f"Selected agent: {result['selected_agent']}")
        print(f"Selected tools: {', '.join(result.get('selected_tools', []))}")
        print(f"Reward: {result['reward']:.4f}")
    
    # Generate policy explanation
    explanation = await policy_explainer.explain_policy(
        agent_name="advanced_rl_coordinator",
        policy_type="deep_rl",
        policy_data=db.get_drl_weights("advanced_rl_coordinator_deep_rl") or {}
    )
    
    print("\nPolicy Explanation:")
    print(explanation)


async def demonstrate_multi_objective_rl() -> None:
    """Demonstrate multi-objective reinforcement learning."""
    print("\n=== Demonstrating Multi-Objective Reinforcement Learning ===\n")
    
    # Create mock tools
    mock_tools = await create_mock_tools()
    
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mock_tools)
    
    # Create multi-objective RL coordinator agent
    objectives = ["user_satisfaction", "task_completion", "efficiency", "accuracy"]
    mo_rl_coordinator = await create_multi_objective_rl_agent_architecture(
        model=model,
        db=db,
        sub_agents=sub_agents,
        objectives=objectives
    )
    
    # Process a few requests
    requests = [
        "What is the weather in Tokyo?",
        "Calculate 25 / 5",
        "Translate 'thank you' to German",
        "Search for information about multi-objective reinforcement learning",
        "Summarize this book on artificial intelligence",
    ]
    
    for request in requests:
        print(f"\nProcessing request: {request}")
        result = await mo_rl_coordinator.process_request(request, [])
        print(f"Selected agent: {result['selected_agent']}")
        
        # Print rewards for each objective
        rewards = result["rewards"]
        print(f"Total reward: {rewards['total']:.4f}")
        for objective in objectives:
            if objective in rewards:
                print(f"{objective.capitalize()} reward: {rewards[objective]:.4f}")
    
    # Get multi-objective Q-value summary
    state = "translation_query"  # Example state
    mo_q_value_summary = q_value_visualizer.get_multi_objective_q_value_summary(
        "mo_rl_coordinator_moql", state, list(sub_agents.keys()), objectives
    )
    
    print("\nMulti-Objective Q-Value Summary:")
    print(f"State: {mo_q_value_summary['state']}")
    print("Best actions by objective:")
    for objective, action in mo_q_value_summary["best_actions"].items():
        print(f"- {objective}: {action}")


async def demonstrate_ab_testing() -> None:
    """Demonstrate A/B testing of different RL strategies."""
    print("\n=== Demonstrating A/B Testing of RL Strategies ===\n")
    
    # Create mock tools
    mock_tools = await create_mock_tools()
    
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mock_tools)
    
    # Create A/B testing framework
    ab_testing_framework = RLABTestingFramework(
        model=model,
        db=db,
        sub_agents=sub_agents,
        tools=mock_tools,
        exploration_rate=0.3
    )
    
    # Add variants
    await ab_testing_framework.add_variant(
        name="basic_q_learning",
        variant_type="basic",
        config={"rl_agent_type": "q_learning"},
        set_as_default=True
    )
    
    await ab_testing_framework.add_variant(
        name="advanced_deep_rl",
        variant_type="advanced",
        config={"rl_agent_type": "deep_rl"}
    )
    
    await ab_testing_framework.add_variant(
        name="multi_objective",
        variant_type="multi_objective",
        config={"objectives": ["user_satisfaction", "task_completion", "efficiency", "accuracy"]}
    )
    
    # Process a set of requests
    requests = [
        "What is the weather in Paris?",
        "Calculate 12 * 12",
        "Translate 'good morning' to Italian",
        "Search for information about A/B testing",
        "Summarize this article about machine learning",
        "What is the weather in Berlin?",
        "Calculate the cube root of 27",
        "Translate 'how are you' to Japanese",
        "Search for information about natural language processing",
        "Summarize this paper on computer vision",
    ]
    
    for request in requests:
        print(f"\nProcessing request: {request}")
        result = await ab_testing_framework.process_request(request, [])
        print(f"Selected variant: {result['variant']}")
        print(f"Selected agent: {result['selected_agent']}")
        if "selected_tools" in result:
            print(f"Selected tools: {', '.join(result['selected_tools'])}")
        
        # Handle different reward formats
        reward = result.get("reward", 0.0)
        if isinstance(reward, dict) and "total" in reward:
            print(f"Total reward: {reward['total']:.4f}")
        else:
            print(f"Reward: {reward:.4f}")
    
    # Get test results
    test_results = ab_testing_framework.get_test_results()
    
    print("\nA/B Testing Results:")
    print(f"Total requests: {test_results['total_requests']}")
    
    print("\nVariant Summaries:")
    for name, summary in test_results["variant_summaries"].items():
        print(f"- {name}:")
        print(f"  - Success rate: {summary['success_rate']:.4f}")
        print(f"  - Average reward: {summary['avg_reward']:.4f}")
        print(f"  - Average response time: {summary['avg_response_time']:.4f}s")
        print(f"  - Requests: {summary['total_requests']}")
    
    print("\nBest Variants:")
    print(f"- By success rate: {test_results['best_variants']['by_success_rate']}")
    print(f"- By average reward: {test_results['best_variants']['by_avg_reward']}")
    print(f"- By response time: {test_results['best_variants']['by_response_time']}")
    
    # Auto-optimize
    best_variant = ab_testing_framework.auto_optimize()
    print(f"\nAuto-selected best variant: {best_variant}")


async def main() -> None:
    """Main function."""
    print("=== Advanced RL Decision-Making Example ===")
    
    # Demonstrate basic RL
    await demonstrate_basic_rl()
    
    # Demonstrate advanced RL
    await demonstrate_advanced_rl()
    
    # Demonstrate multi-objective RL
    await demonstrate_multi_objective_rl()
    
    # Demonstrate A/B testing
    await demonstrate_ab_testing()
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
