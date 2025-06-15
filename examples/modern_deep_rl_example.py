"""
Example script demonstrating modern deep reinforcement learning capabilities.
This example shows how to use DQN, PPO, A2C, and Rainbow DQN algorithms.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.agents.advanced_rl_techniques import RainbowDQNAgent
from src.agents.enhanced_state_representation import (
    ContextualStateEncoder,
    TextEmbeddingEncoder,
)
from src.agents.modern_deep_rl import (
    DQNAgent,
    PPOAgent,
    create_modern_deep_rl_agent_architecture,
)
from src.agents.reinforcement_learning import RewardSystem
from src.memory.memory_persistence import MemoryDatabase

# Load environment variables
load_dotenv()


async def demonstrate_dqn_agent():
    """Demonstrate DQN agent capabilities."""
    print("\n🎯 Demonstrating DQN Agent")
    print("=" * 50)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("dqn_demo.db")
    reward_system = RewardSystem(db)

    # Create DQN agent
    dqn_agent = DQNAgent(
        name="dqn_demo",
        model=model,
        db=db,
        reward_system=reward_system,
        state_dim=128,
        action_dim=5,
        learning_rate=1e-4,
        epsilon=1.0,
        epsilon_decay=0.995,
        double_dqn=True,
        dueling=True,
        prioritized_replay=True,
    )

    print("✅ Created DQN agent with:")
    print(f"   - Double DQN: {dqn_agent.double_dqn}")
    print(f"   - Dueling architecture: {dqn_agent.q_network.dueling}")
    print(f"   - Prioritized replay: {dqn_agent.replay_buffer.prioritized}")
    print(f"   - Initial epsilon: {dqn_agent.epsilon}")

    # Simulate some training episodes
    print("\n🏋️ Training DQN agent...")
    for episode in range(10):
        state = np.random.randn(128).astype(np.float32)

        for step in range(20):
            # Select action
            action = dqn_agent.select_action(state, training=True)

            # Simulate environment step
            next_state = np.random.randn(128).astype(np.float32)
            reward = np.random.uniform(-1, 1)
            done = (step == 19)

            # Store experience
            dqn_agent.store_experience(state, action, reward, next_state, done)

            # Train if enough experiences
            if len(dqn_agent.replay_buffer) > dqn_agent.batch_size:
                metrics = dqn_agent.train()
                if metrics and step % 10 == 0:
                    print(f"   Episode {episode}, Step {step}: {metrics}")

            state = next_state
            if done:
                break

    print(f"✅ DQN training completed. Final epsilon: {dqn_agent.epsilon:.3f}")


async def demonstrate_ppo_agent():
    """Demonstrate PPO agent capabilities."""
    print("\n🎯 Demonstrating PPO Agent")
    print("=" * 50)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("ppo_demo.db")
    reward_system = RewardSystem(db)

    # Create PPO agent
    ppo_agent = PPOAgent(
        name="ppo_demo",
        model=model,
        db=db,
        reward_system=reward_system,
        state_dim=128,
        action_dim=5,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        ppo_epochs=4,
        continuous=False,
    )

    print("✅ Created PPO agent with:")
    print(f"   - Clip epsilon: {ppo_agent.clip_epsilon}")
    print(f"   - PPO epochs: {ppo_agent.ppo_epochs}")
    print(f"   - Continuous actions: {ppo_agent.continuous}")

    # Simulate training episode
    print("\n🏋️ Training PPO agent...")
    state = np.random.randn(128).astype(np.float32)

    for step in range(50):
        # Select action
        action, log_prob, value = ppo_agent.select_action(state)

        # Simulate environment step
        next_state = np.random.randn(128).astype(np.float32)
        reward = np.random.uniform(-1, 1)
        done = (step == 49)

        # Store experience
        ppo_agent.store_experience(state, action, log_prob, reward, value, done)

        state = next_state
        if done:
            break

    # Train on collected episode
    metrics = ppo_agent.train()
    print(f"✅ PPO training metrics: {metrics}")


async def demonstrate_rainbow_dqn():
    """Demonstrate Rainbow DQN agent capabilities."""
    print("\n🎯 Demonstrating Rainbow DQN Agent")
    print("=" * 50)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("rainbow_demo.db")
    reward_system = RewardSystem(db)

    # Create Rainbow DQN agent
    rainbow_agent = RainbowDQNAgent(
        name="rainbow_demo",
        model=model,
        db=db,
        reward_system=reward_system,
        state_dim=128,
        action_dim=5,
        multi_step=3,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
    )

    print("✅ Created Rainbow DQN agent with:")
    print(f"   - Multi-step learning: {rainbow_agent.multi_step}")
    print(f"   - Distributional RL atoms: {rainbow_agent.num_atoms}")
    print(f"   - Value range: [{rainbow_agent.v_min}, {rainbow_agent.v_max}]")
    print(f"   - Noisy networks: {rainbow_agent.q_network.noisy}")
    print(f"   - Dueling architecture: {rainbow_agent.q_network.dueling}")

    # Simulate training
    print("\n🏋️ Training Rainbow DQN agent...")
    for episode in range(5):
        state = np.random.randn(128).astype(np.float32)

        for step in range(30):
            # Select action (no epsilon needed due to noisy networks)
            action = rainbow_agent.select_action(state, training=True)

            # Simulate environment step
            next_state = np.random.randn(128).astype(np.float32)
            reward = np.random.uniform(-1, 1)
            done = (step == 29)

            # Store experience (handles multi-step automatically)
            rainbow_agent.store_experience(state, action, reward, next_state, done)

            # Train if enough experiences
            if len(rainbow_agent.replay_buffer) > rainbow_agent.batch_size:
                metrics = rainbow_agent.train()
                if metrics and step % 15 == 0:
                    print(f"   Episode {episode}, Step {step}: {metrics}")

            state = next_state
            if done:
                break

    print("✅ Rainbow DQN training completed!")


async def demonstrate_enhanced_state_representation():
    """Demonstrate enhanced state representation."""
    print("\n🎯 Demonstrating Enhanced State Representation")
    print("=" * 50)

    # Create text encoder
    text_encoder = TextEmbeddingEncoder(model_name="all-MiniLM-L6-v2")

    # Create contextual state encoder
    state_encoder = ContextualStateEncoder(
        text_encoder=text_encoder,
        include_temporal=True,
        include_performance=True,
        include_user_profile=True,
    )

    print("✅ Created contextual state encoder with:")
    print(f"   - Text embedding dimension: {state_encoder.text_dim}")
    print(f"   - Temporal features: {state_encoder.temporal_dim}")
    print(f"   - Performance features: {state_encoder.performance_dim}")
    print(f"   - User profile features: {state_encoder.user_profile_dim}")
    print(f"   - Total dimension: {state_encoder.total_dim}")

    # Create sample context
    context = {
        "request": "Can you help me analyze this data and create a visualization?",
        "history": [
            {"role": "user", "content": "Hello, I need help with data analysis"},
            {"role": "assistant", "content": "I'd be happy to help with data analysis!"},
        ],
        "recent_rewards": [0.8, 0.6, 0.9, 0.7],
        "recent_response_times": [1.2, 0.8, 1.5, 1.0],
        "tool_usage_counts": {"search": 5, "analyze": 3, "visualize": 2},
        "user_profile": {
            "preferences": {"verbosity": 0.7, "technical_level": 0.8},
            "expertise": {"technology": 0.9, "business": 0.6},
        },
    }

    # Encode state
    db = MemoryDatabase("state_demo.db")
    state_vector = await state_encoder.encode_state(context, db)

    print("\n🧠 Encoded state vector:")
    print(f"   - Shape: {state_vector.shape}")
    print(f"   - Data type: {state_vector.dtype}")
    print(f"   - Sample values: {state_vector[:10]}")

    print("✅ State encoding completed!")


async def demonstrate_modern_deep_rl_coordinator():
    """Demonstrate the modern deep RL coordinator."""
    print("\n🎯 Demonstrating Modern Deep RL Coordinator")
    print("=" * 50)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("coordinator_demo.db")

    # Create mock sub-agents and tools
    sub_agents = {
        "search_agent": type("MockAgent", (), {
            "process_request": lambda self, req, hist: asyncio.create_task(
                asyncio.coroutine(lambda: {"success": True, "response": f"Searched for: {req}"})()
            )
        })(),
        "analysis_agent": type("MockAgent", (), {
            "process_request": lambda self, req, hist: asyncio.create_task(
                asyncio.coroutine(lambda: {"success": True, "response": f"Analyzed: {req}"})()
            )
        })(),
    }

    tools = [
        type("MockTool", (), {"name": "calculator", "arun": lambda self, x: f"Calculated: {x}"})(),
        type("MockTool", (), {"name": "translator", "arun": lambda self, x: f"Translated: {x}"})(),
    ]

    # Create coordinator with DQN
    coordinator = await create_modern_deep_rl_agent_architecture(
        model=model,
        db=db,
        sub_agents=sub_agents,
        tools=tools,
        rl_algorithm="dqn",
        double_dqn=True,
        dueling=True,
    )

    print("✅ Created coordinator with:")
    print(f"   - RL algorithm: {coordinator.rl_algorithm}")
    print(f"   - Available actions: {len(coordinator.actions)}")
    print(f"   - State dimension: {coordinator.state_dim}")

    # Simulate some interactions
    print("\n🤖 Simulating interactions...")
    requests = [
        "Search for information about machine learning",
        "Analyze the search results",
        "Calculate the average performance",
        "Translate this text to Spanish",
    ]

    for i, request in enumerate(requests):
        print(f"\n📝 Request {i+1}: {request}")

        result = await coordinator.process_request(request, [])

        print(f"   ✅ Success: {result['success']}")
        print(f"   🎯 Selected action: {result['selected_action']}")
        print(f"   🏆 Reward: {result['reward']:.3f}")

        # Train after each interaction
        training_metrics = await coordinator.train_episode()
        if training_metrics:
            print(f"   📈 Training: {training_metrics}")

    print("✅ Coordinator demonstration completed!")


async def main():
    """Run all demonstrations."""
    print("🚀 Modern Deep RL Demonstration")
    print("=" * 60)

    try:
        # Import numpy here to avoid issues
        import numpy as np
        globals()['np'] = np

        await demonstrate_enhanced_state_representation()
        await demonstrate_dqn_agent()
        await demonstrate_ppo_agent()
        await demonstrate_rainbow_dqn()
        await demonstrate_modern_deep_rl_coordinator()

        print("\n🎉 All demonstrations completed successfully!")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Please install required packages:")
        print("   pip install torch sentence-transformers numpy")
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
