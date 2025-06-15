"""
Reinforcement learning entry point for DataMCPServerAgent.
This version implements a sophisticated reinforcement learning system for continuous improvement
with advanced decision-making capabilities.
"""

import asyncio
import os
from typing import Any, List, Union

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.advanced_rl_decision_making import (
    AdvancedRLCoordinatorAgent,
    create_advanced_rl_agent_architecture,
)
from src.agents.advanced_rl_techniques import RainbowDQNAgent
from src.agents.agent_architecture import create_specialized_sub_agents
from src.agents.curriculum_learning import create_curriculum_learning_agent
from src.agents.distributed_rl import create_distributed_rl_system
from src.agents.explainable_rl import create_explainable_rl_agent
from src.agents.hierarchical_rl import (
    HierarchicalRLCoordinatorAgent,
    create_hierarchical_rl_agent_architecture,
)
from src.agents.meta_learning_rl import MAMLAgent
from src.agents.modern_deep_rl import (
    ModernDeepRLCoordinatorAgent,
    create_modern_deep_rl_agent_architecture,
)
from src.agents.multi_agent_rl import create_multi_agent_rl_architecture
from src.agents.multi_objective_rl import (
    MultiObjectiveRLCoordinatorAgent,
    create_multi_objective_rl_agent_architecture,
)
from src.agents.reinforcement_learning import (
    RLCoordinatorAgent,
    create_rl_agent_architecture,
)
from src.agents.safe_rl import ResourceUsageConstraint, ResponseTimeConstraint, create_safe_rl_agent
from src.memory.advanced_memory_persistence import (
    AdvancedMemoryDatabase as MemoryDatabase,
)
from src.memory.hierarchical_memory_persistence import HierarchicalMemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.decision_explanation import DecisionExplainer, PolicyExplainer
from src.utils.error_handlers import format_error_for_user
from src.utils.rl_ab_testing import RLABTestingFramework

load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize memory database
db_path = os.getenv("MEMORY_DB_PATH", "agent_memory.db")
db = MemoryDatabase(db_path)

# Initialize decision explainer
decision_explainer = DecisionExplainer(
    model=model, db=db, explanation_level=os.getenv("EXPLANATION_LEVEL", "moderate")
)

# Initialize policy explainer
policy_explainer = PolicyExplainer(model=model, db=db)


async def setup_rl_agent(mcp_tools: List[BaseTool], rl_mode: str = "auto") -> Union[
    RLCoordinatorAgent,
    AdvancedRLCoordinatorAgent,
    MultiObjectiveRLCoordinatorAgent,
    HierarchicalRLCoordinatorAgent,
    ModernDeepRLCoordinatorAgent,
    RainbowDQNAgent,
    Any,  # For new agent types
]:
    """Set up the reinforcement learning agent.

    Args:
        mcp_tools: List of MCP tools
        rl_mode: RL mode to use ("basic", "advanced", "multi_objective",
                "hierarchical", "modern_deep", "rainbow", "multi_agent",
                "curriculum", "meta_learning", "distributed", "safe",
                "explainable", or "auto")

    Returns:
        RL coordinator agent
    """
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mcp_tools)

    # Determine RL mode if auto
    if rl_mode == "auto":
        rl_mode = os.getenv("RL_MODE", "advanced")

    # Create appropriate RL agent based on mode
    if rl_mode == "basic":
        # Create basic RL coordinator agent
        rl_coordinator = await create_rl_agent_architecture(
            model=model,
            db=db,
            sub_agents=sub_agents,
            rl_agent_type=os.getenv("RL_AGENT_TYPE", "q_learning"),
        )
    elif rl_mode == "advanced":
        # Create advanced RL coordinator agent
        rl_coordinator = await create_advanced_rl_agent_architecture(
            model=model,
            db=db,
            sub_agents=sub_agents,
            tools=mcp_tools,
            rl_agent_type=os.getenv("RL_AGENT_TYPE", "deep_rl"),
        )
    elif rl_mode == "multi_objective":
        # Create multi-objective RL coordinator agent
        objectives = os.getenv(
            "RL_OBJECTIVES", "user_satisfaction,task_completion,efficiency,accuracy"
        ).split(",")
        rl_coordinator = await create_multi_objective_rl_agent_architecture(
            model=model, db=db, sub_agents=sub_agents, objectives=objectives
        )
    elif rl_mode == "hierarchical":
        # Create hierarchical RL coordinator agent
        # Convert the regular memory database to hierarchical memory database
        hierarchical_db = HierarchicalMemoryDatabase(db_path)
        rl_coordinator = await create_hierarchical_rl_agent_architecture(
            model=model,
            db=hierarchical_db,
            sub_agents=sub_agents,
            tools=mcp_tools,
        )
    elif rl_mode == "modern_deep":
        # Create modern deep RL coordinator agent
        rl_algorithm = os.getenv("RL_ALGORITHM", "dqn")
        rl_coordinator = await create_modern_deep_rl_agent_architecture(
            model=model,
            db=db,
            sub_agents=sub_agents,
            tools=mcp_tools,
            rl_algorithm=rl_algorithm,
            double_dqn=os.getenv("DQN_DOUBLE", "true").lower() == "true",
            dueling=os.getenv("DQN_DUELING", "true").lower() == "true",
            prioritized_replay=os.getenv("DQN_PRIORITIZED_REPLAY", "true").lower() == "true",
        )
    elif rl_mode == "rainbow":
        # Create Rainbow DQN agent
        from src.agents.reinforcement_learning import RewardSystem
        reward_system = RewardSystem(db)

        rl_coordinator = RainbowDQNAgent(
            name="rainbow_dqn_coordinator",
            model=model,
            db=db,
            reward_system=reward_system,
            state_dim=int(os.getenv("RAINBOW_STATE_DIM", "512")),
            action_dim=len(sub_agents) + len(mcp_tools),
            multi_step=int(os.getenv("RAINBOW_MULTI_STEP", "3")),
            num_atoms=int(os.getenv("RAINBOW_NUM_ATOMS", "51")),
            v_min=float(os.getenv("RAINBOW_V_MIN", "-10.0")),
            v_max=float(os.getenv("RAINBOW_V_MAX", "10.0")),
        )
    elif rl_mode == "multi_agent":
        # Create multi-agent RL coordinator
        num_agents = int(os.getenv("MULTI_AGENT_COUNT", "3"))
        cooperation_mode = os.getenv("MULTI_AGENT_MODE", "cooperative")
        communication = os.getenv("MULTI_AGENT_COMMUNICATION", "true").lower() == "true"

        rl_coordinator = await create_multi_agent_rl_architecture(
            model=model,
            db=db,
            num_agents=num_agents,
            state_dim=int(os.getenv("MULTI_AGENT_STATE_DIM", "128")),
            action_dim=len(sub_agents) + len(mcp_tools),
            cooperation_mode=cooperation_mode,
            communication=communication,
        )
    elif rl_mode == "curriculum":
        # Create curriculum learning agent
        base_rl_mode = os.getenv("CURRICULUM_BASE_RL", "dqn")

        # Create base agent first
        if base_rl_mode == "dqn":
            from src.agents.modern_deep_rl import DQNAgent
            from src.agents.reinforcement_learning import RewardSystem
            reward_system = RewardSystem(db)

            base_agent = DQNAgent(
                name="curriculum_base_dqn",
                model=model,
                db=db,
                reward_system=reward_system,
                state_dim=int(os.getenv("CURRICULUM_STATE_DIM", "128")),
                action_dim=len(sub_agents) + len(mcp_tools),
            )
        else:
            # Default to basic RL agent
            base_agent = await create_rl_agent_architecture(
                model=model, db=db, sub_agents=sub_agents, tools=mcp_tools
            )

        rl_coordinator = await create_curriculum_learning_agent(
            model=model,
            db=db,
            base_agent=base_agent,
            difficulty_increment=float(os.getenv("CURRICULUM_DIFFICULTY_INCREMENT", "0.1")),
        )
    elif rl_mode == "meta_learning":
        # Create meta-learning agent (MAML)
        from src.agents.reinforcement_learning import RewardSystem
        reward_system = RewardSystem(db)

        rl_coordinator = MAMLAgent(
            name="maml_coordinator",
            model=model,
            db=db,
            reward_system=reward_system,
            state_dim=int(os.getenv("MAML_STATE_DIM", "128")),
            action_dim=len(sub_agents) + len(mcp_tools),
            meta_lr=float(os.getenv("MAML_META_LR", "1e-3")),
            inner_lr=float(os.getenv("MAML_INNER_LR", "1e-2")),
            inner_steps=int(os.getenv("MAML_INNER_STEPS", "5")),
        )
    elif rl_mode == "distributed":
        # Create distributed RL system
        num_workers = int(os.getenv("DISTRIBUTED_WORKERS", "4"))
        model_type = os.getenv("DISTRIBUTED_MODEL_TYPE", "dqn")

        rl_coordinator = await create_distributed_rl_system(
            model=model,
            db=db,
            num_workers=num_workers,
            model_type=model_type,
            state_dim=int(os.getenv("DISTRIBUTED_STATE_DIM", "128")),
            action_dim=len(sub_agents) + len(mcp_tools),
        )
    elif rl_mode == "safe":
        # Create safe RL agent
        base_rl_mode = os.getenv("SAFE_BASE_RL", "dqn")

        # Create base agent first
        if base_rl_mode == "dqn":
            from src.agents.modern_deep_rl import DQNAgent
            from src.agents.reinforcement_learning import RewardSystem
            reward_system = RewardSystem(db)

            base_agent = DQNAgent(
                name="safe_base_dqn",
                model=model,
                db=db,
                reward_system=reward_system,
                state_dim=int(os.getenv("SAFE_STATE_DIM", "128")),
                action_dim=len(sub_agents) + len(mcp_tools),
            )
        else:
            # Default to basic RL agent
            base_agent = await create_rl_agent_architecture(
                model=model, db=db, sub_agents=sub_agents, tools=mcp_tools
            )

        # Create safety constraints
        safety_constraints = [
            ResourceUsageConstraint(
                max_resource_usage=float(os.getenv("SAFE_MAX_RESOURCE_USAGE", "0.8"))
            ),
            ResponseTimeConstraint(
                max_response_time=float(os.getenv("SAFE_MAX_RESPONSE_TIME", "5.0"))
            ),
        ]

        rl_coordinator = await create_safe_rl_agent(
            model=model,
            db=db,
            base_agent=base_agent,
            safety_constraints=safety_constraints,
            safety_weight=float(os.getenv("SAFE_WEIGHT", "0.5")),
        )
    elif rl_mode == "explainable":
        # Create explainable RL agent
        base_rl_mode = os.getenv("EXPLAINABLE_BASE_RL", "dqn")

        # Create base agent first
        if base_rl_mode == "dqn":
            from src.agents.modern_deep_rl import DQNAgent
            from src.agents.reinforcement_learning import RewardSystem
            reward_system = RewardSystem(db)

            base_agent = DQNAgent(
                name="explainable_base_dqn",
                model=model,
                db=db,
                reward_system=reward_system,
                state_dim=int(os.getenv("EXPLAINABLE_STATE_DIM", "128")),
                action_dim=len(sub_agents) + len(mcp_tools),
            )
        else:
            # Default to basic RL agent
            base_agent = await create_rl_agent_architecture(
                model=model, db=db, sub_agents=sub_agents, tools=mcp_tools
            )

        # Define feature names
        feature_names = os.getenv("EXPLAINABLE_FEATURE_NAMES", "").split(",")
        if not feature_names or feature_names == [""]:
            feature_names = None

        explanation_methods = os.getenv("EXPLAINABLE_METHODS", "gradient,permutation").split(",")

        rl_coordinator = await create_explainable_rl_agent(
            model=model,
            db=db,
            base_agent=base_agent,
            feature_names=feature_names,
            explanation_methods=explanation_methods,
        )
    else:
        raise ValueError(f"Unknown RL mode: {rl_mode}")

    return rl_coordinator


async def chat_with_rl_agent() -> None:
    """Chat with the reinforcement learning agent."""
    # Set up the MCP server parameters
    server_params = StdioServerParameters(
        command="npx",
        env={
            "API_TOKEN": os.getenv("API_TOKEN"),
            "BROWSER_AUTH": os.getenv("BROWSER_AUTH"),
            "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
        },
        args=["@brightdata/mcp"],
    )

    # Create the MCP client session
    async with stdio_client(server_params) as client:
        # Create the MCP session
        async with ClientSession(client) as session:
            # Load MCP tools
            mcp_tools = load_mcp_tools(session)

            # Add Bright Data tools
            bright_data_toolkit = BrightDataToolkit(session)
            mcp_tools.extend(bright_data_toolkit.get_tools())

            # Set up the RL agent
            rl_mode = os.getenv("RL_MODE", "advanced")
            rl_agent = await setup_rl_agent(mcp_tools, rl_mode)

            # Set up A/B testing if enabled
            ab_testing_enabled = os.getenv("RL_AB_TESTING_ENABLED", "false").lower() == "true"
            ab_testing_framework = None

            if ab_testing_enabled:
                print("\n=== Setting up A/B Testing Framework ===")
                ab_testing_framework = RLABTestingFramework(
                    model=model,
                    db=db,
                    sub_agents=await create_specialized_sub_agents(model, mcp_tools),
                    tools=mcp_tools,
                    exploration_rate=float(os.getenv("RL_AB_TESTING_EXPLORATION_RATE", "0.2")),
                )

                # Add variants
                await ab_testing_framework.add_variant(
                    name="basic_q_learning",
                    variant_type="basic",
                    config={"rl_agent_type": "q_learning"},
                    set_as_default=True,
                )

                await ab_testing_framework.add_variant(
                    name="advanced_deep_rl",
                    variant_type="advanced",
                    config={"rl_agent_type": "deep_rl"},
                )

                await ab_testing_framework.add_variant(
                    name="multi_objective",
                    variant_type="multi_objective",
                    config={
                        "objectives": [
                            "user_satisfaction",
                            "task_completion",
                            "efficiency",
                            "accuracy",
                        ]
                    },
                )

                print("A/B Testing Framework initialized with 3 variants.")

            print(f"\n=== Advanced Reinforcement Learning Agent ({rl_mode.upper()} mode) ===\n")
            print("Commands:")
            print("- 'exit': Quit the application")
            print("- 'feedback: <message>': Provide feedback on the last response")
            print("- 'learn': Perform batch learning")
            print("- 'explain': Get an explanation of the last decision")
            print("- 'policy': Get an explanation of the current policy")
            if ab_testing_enabled:
                print("- 'ab_results': Get A/B testing results")
                print("- 'optimize': Automatically select the best variant")

            # Initialize conversation history
            history = []
            last_result = None

            # Main chat loop
            while True:
                # Get user input
                user_input = input("\nYou: ")

                # Check for exit command
                if user_input.lower() == "exit":
                    break

                # Check for feedback command
                if user_input.lower().startswith("feedback:"):
                    if not history:
                        print("No conversation to provide feedback for.")
                        continue

                    feedback = user_input[len("feedback:") :].strip()

                    # Get the last request and response
                    last_request = next(
                        (msg["content"] for msg in reversed(history) if msg["role"] == "user"),
                        None,
                    )
                    last_response = next(
                        (msg["content"] for msg in reversed(history) if msg["role"] == "assistant"),
                        None,
                    )

                    if last_request and last_response:
                        # Update from feedback
                        await rl_agent.update_from_feedback(last_request, last_response, feedback)

                        # Save interaction for batch learning
                        db.save_agent_interaction(
                            "rl_coordinator", last_request, last_response, feedback
                        )

                        print("Feedback recorded. Thank you!")
                    else:
                        print("No conversation to provide feedback for.")

                    continue

                # Check for learn command
                if user_input.lower() == "learn":
                    print("\nPerforming batch learning...")
                    learning_result = await rl_agent.learn_from_batch()
                    print(f"Learning result: {learning_result}")
                    continue

                # Check for explain command
                if user_input.lower() == "explain":
                    if last_result is None:
                        print("No decision to explain yet.")
                        continue

                    print("\nGenerating decision explanation...")

                    # Extract information for explanation
                    context = {
                        "request": last_result.get("request", ""),
                        "history": history[-4:],
                    }
                    selected_action = last_result.get("selected_agent", "")
                    alternative_actions = [
                        agent for agent in rl_agent.sub_agents.keys() if agent != selected_action
                    ]

                    # Get state and q-values
                    if hasattr(rl_agent, "rl_agent") and hasattr(rl_agent.rl_agent, "q_table"):
                        # For Q-learning
                        state = (
                            await rl_agent._extract_state(context)
                            if hasattr(rl_agent, "_extract_state")
                            else "unknown"
                        )
                        q_values = rl_agent.rl_agent.q_table.get(state, {})
                    else:
                        # For other RL types
                        state = "unknown"
                        q_values = {}

                    # Generate explanation
                    explanation = await decision_explainer.explain_decision(
                        context=context,
                        selected_action=selected_action,
                        alternative_actions=alternative_actions,
                        state=state,
                        q_values=q_values,
                    )

                    print(f"\nDecision Explanation:\n{explanation}")
                    continue

                # Check for policy explanation command
                if user_input.lower() == "policy":
                    print("\nGenerating policy explanation...")

                    # Determine policy type
                    if rl_mode == "basic":
                        policy_type = "q_learning"
                        policy_data = db.get_q_table("rl_coordinator_q_learning") or {}
                    elif rl_mode == "advanced":
                        policy_type = "deep_rl"
                        policy_data = db.get_drl_weights("advanced_rl_coordinator_deep_rl") or {}
                    elif rl_mode == "multi_objective":
                        policy_type = "multi_objective"
                        policy_data = db.get_mo_q_tables("mo_rl_coordinator_moql") or {}
                    else:
                        policy_type = "unknown"
                        policy_data = {}

                    # Generate explanation
                    explanation = await policy_explainer.explain_policy(
                        agent_name="rl_coordinator",
                        policy_type=policy_type,
                        policy_data=policy_data,
                    )

                    print(f"\nPolicy Explanation:\n{explanation}")
                    continue

                # Check for A/B testing results command
                if user_input.lower() == "ab_results" and ab_testing_enabled:
                    if ab_testing_framework is None:
                        print("A/B testing framework not initialized.")
                        continue

                    print("\nGenerating A/B testing results...")
                    results = ab_testing_framework.get_test_results()

                    print("\nA/B Testing Results:")
                    print(f"Total requests: {results['total_requests']}")
                    print("\nVariant Summaries:")

                    for name, summary in results["variant_summaries"].items():
                        print(f"- {name}:")
                        print(f"  - Success rate: {summary['success_rate']:.4f}")
                        print(f"  - Average reward: {summary['avg_reward']:.4f}")
                        print(f"  - Average response time: {summary['avg_response_time']:.4f}s")
                        print(f"  - Requests: {summary['total_requests']}")

                    print("\nBest Variants:")
                    print(f"- By success rate: {results['best_variants']['by_success_rate']}")
                    print(f"- By average reward: {results['best_variants']['by_avg_reward']}")
                    print(f"- By response time: {results['best_variants']['by_response_time']}")

                    continue

                # Check for optimize command
                if user_input.lower() == "optimize" and ab_testing_enabled:
                    if ab_testing_framework is None:
                        print("A/B testing framework not initialized.")
                        continue

                    print("\nOptimizing RL strategy...")
                    best_variant = ab_testing_framework.auto_optimize()

                    print(f"Selected best variant: {best_variant}")
                    continue

                try:
                    # Process the request
                    if ab_testing_enabled and ab_testing_framework is not None:
                        # Use A/B testing framework
                        result = await ab_testing_framework.process_request(user_input, history)
                        print(f"[Debug] Using variant: {result['variant']}")
                    else:
                        # Use regular RL agent
                        result = await rl_agent.process_request(user_input, history)

                    # Store the result for later explanation
                    result["request"] = user_input
                    last_result = result

                    # Print the response
                    if result["success"]:
                        print(f"\nAgent: {result['response']}")
                    else:
                        print(f"\nAgent Error: {result['error']}")

                    # Add to history
                    history.append({"role": "user", "content": user_input})
                    history.append(
                        {
                            "role": "assistant",
                            "content": result["response"] if result["success"] else result["error"],
                        }
                    )

                    # Print debug info
                    print(f"\n[Debug] Selected agent: {result['selected_agent']}")

                    # Handle different reward formats
                    reward = result.get("reward", 0.0)
                    if isinstance(reward, dict) and "total" in reward:
                        print(f"[Debug] Reward: {reward['total']:.4f}")
                        # Print individual objective rewards for multi-objective RL
                        for obj, val in reward.items():
                            if obj != "total":
                                print(f"[Debug] {obj.capitalize()} reward: {val:.4f}")
                    else:
                        print(f"[Debug] Reward: {reward:.4f}")

                    # Print selected tools if available
                    if "selected_tools" in result:
                        print(f"[Debug] Selected tools: {', '.join(result['selected_tools'])}")

                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"\nError: {error_message}")


if __name__ == "__main__":
    asyncio.run(chat_with_rl_agent())
