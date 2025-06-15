"""
Example demonstrating advanced reinforcement learning features.
This example shows meta-learning, multi-agent RL, curriculum learning, and advanced memory.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.agents.curriculum_learning import create_curriculum_learning_agent
from src.agents.meta_learning_rl import FewShotLearningAgent, MAMLAgent
from src.agents.modern_deep_rl import DQNAgent
from src.agents.multi_agent_rl import create_multi_agent_rl_architecture
from src.agents.reinforcement_learning import RewardSystem
from src.memory.advanced_rl_memory import AdvancedRLMemorySystem
from src.memory.memory_persistence import MemoryDatabase

# Load environment variables
load_dotenv()


async def demonstrate_meta_learning():
    """Demonstrate meta-learning capabilities."""
    print("\n🧠 Demonstrating Meta-Learning (MAML)")
    print("=" * 60)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("meta_learning_demo.db")
    reward_system = RewardSystem(db)

    # Create MAML agent
    maml_agent = MAMLAgent(
        name="maml_demo",
        model=model,
        db=db,
        reward_system=reward_system,
        state_dim=64,
        action_dim=4,
        meta_lr=1e-3,
        inner_lr=1e-2,
        inner_steps=5,
    )

    print("✅ Created MAML agent with:")
    print(f"   - Meta learning rate: {maml_agent.meta_lr}")
    print(f"   - Inner learning rate: {maml_agent.inner_lr}")
    print(f"   - Inner steps: {maml_agent.inner_steps}")

    # Simulate multiple tasks for meta-learning
    print("\n🏋️ Training on multiple tasks...")

    for task_id in range(5):
        # Generate task data
        support_data = []
        query_data = []

        for _ in range(10):  # Support set
            state = np.random.randn(64).astype(np.float32)
            action = np.random.randint(0, 4)
            reward = np.random.uniform(-1, 1)
            support_data.append((
                torch.FloatTensor(state),
                torch.LongTensor([action]),
                reward
            ))

        for _ in range(5):  # Query set
            state = np.random.randn(64).astype(np.float32)
            action = np.random.randint(0, 4)
            reward = np.random.uniform(-1, 1)
            query_data.append((
                torch.FloatTensor(state),
                torch.LongTensor([action]),
                reward
            ))

        # Add task to buffer
        maml_agent.add_task({
            "task_id": f"task_{task_id}",
            "support_data": support_data,
            "query_data": query_data,
        })

    # Train meta-learning
    metrics = maml_agent.train_meta_learning()
    print(f"✅ Meta-learning training completed: {metrics}")

    # Demonstrate fast adaptation
    print("\n🚀 Demonstrating fast adaptation to new task...")
    new_task_data = []
    for _ in range(3):  # Few-shot examples
        state = np.random.randn(64).astype(np.float32)
        action = np.random.randint(0, 4)
        reward = np.random.uniform(0.5, 1.0)  # Positive rewards for new task
        new_task_data.append((
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            reward
        ))

    adapted_network = maml_agent.adapt_to_task(new_task_data)
    print("✅ Successfully adapted to new task with few examples!")


async def demonstrate_multi_agent_rl():
    """Demonstrate multi-agent reinforcement learning."""
    print("\n🤝 Demonstrating Multi-Agent RL")
    print("=" * 60)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("multi_agent_demo.db")

    # Create multi-agent coordinator
    coordinator = await create_multi_agent_rl_architecture(
        model=model,
        db=db,
        num_agents=3,
        state_dim=32,
        action_dim=4,
        cooperation_mode="cooperative",
        communication=True,
    )

    print("✅ Created multi-agent system with:")
    print(f"   - Number of agents: {coordinator.num_agents}")
    print(f"   - Cooperation mode: {coordinator.cooperation_mode}")
    print(f"   - Communication enabled: {coordinator.communication}")

    # Simulate multi-agent interactions
    print("\n🤖 Simulating multi-agent interactions...")

    requests = [
        "Analyze market trends and create investment strategy",
        "Research competitors and develop marketing plan",
        "Optimize resource allocation across departments",
        "Coordinate project timeline and deliverables",
    ]

    for i, request in enumerate(requests):
        print(f"\n📝 Request {i+1}: {request}")

        result = await coordinator.process_multi_agent_request(request, [])

        print(f"   ✅ Success: {result['success']}")
        print(f"   🎯 Actions: {result['actions']}")
        print(f"   🏆 Rewards: {result['rewards']}")
        print(f"   🤝 Cooperation score: {result['cooperation_score']:.3f}")

        # Update target networks periodically
        if i % 2 == 0:
            coordinator.update_target_networks()

    # Get cooperation metrics
    cooperation_metrics = coordinator.get_cooperation_metrics()
    print(f"\n📊 Cooperation Metrics: {cooperation_metrics}")


async def demonstrate_curriculum_learning():
    """Demonstrate curriculum learning."""
    print("\n📚 Demonstrating Curriculum Learning")
    print("=" * 60)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("curriculum_demo.db")
    reward_system = RewardSystem(db)

    # Create base DQN agent
    base_agent = DQNAgent(
        name="base_dqn",
        model=model,
        db=db,
        reward_system=reward_system,
        state_dim=128,
        action_dim=5,
    )

    # Create curriculum learning agent
    curriculum_agent = await create_curriculum_learning_agent(
        model=model,
        db=db,
        base_agent=base_agent,
        difficulty_increment=0.1,
    )

    print("✅ Created curriculum learning agent")
    print(f"   - Curriculum stage: {curriculum_agent.curriculum_stage}")
    print(f"   - Total tasks: {len(curriculum_agent.current_tasks)}")

    # Simulate curriculum learning
    print("\n📖 Simulating curriculum learning...")

    for episode in range(10):
        current_task = curriculum_agent.get_current_task()

        if current_task is None:
            print("🎓 Curriculum completed!")
            break

        print(f"\n📋 Episode {episode + 1}: {current_task.task_id}")
        print(f"   Difficulty: {current_task.difficulty:.2f}")
        print(f"   Description: {current_task.description}")

        # Process task
        context = {
            "history": [],
            "episode": episode,
        }

        result = await curriculum_agent.process_task_request(current_task, context)

        print(f"   ✅ Success: {result['success']}")
        print(f"   🎯 Attempts: {result['attempts']}")
        print(f"   📈 Success rate: {result['success_rate']:.2f}")
        print(f"   🏆 Reward: {result['reward']:.3f}")

        if result['task_mastered']:
            print("   🌟 Task mastered!")

        # Advance curriculum if needed
        await curriculum_agent.advance_curriculum()

    # Get learning progress
    progress = curriculum_agent.get_learning_progress()
    print("\n📊 Learning Progress:")
    print(f"   - Curriculum stage: {progress['curriculum_stage']}")
    print(f"   - Completed tasks: {progress['completed_tasks']}")
    print(f"   - Mastered tasks: {progress['mastered_tasks']}")
    print(f"   - Mastery rate: {progress['mastery_rate']:.2f}")
    print(f"   - Learning velocity: {progress['learning_velocity']:.3f}")


async def demonstrate_advanced_memory():
    """Demonstrate advanced memory systems."""
    print("\n🧠 Demonstrating Advanced Memory Systems")
    print("=" * 60)

    # Initialize components
    db = MemoryDatabase("advanced_memory_demo.db")

    # Create advanced memory system
    memory_system = AdvancedRLMemorySystem(
        db=db,
        state_dim=64,
        action_dim=4,
        episodic_capacity=1000,
        working_memory_capacity=10,
    )

    print("✅ Created advanced memory system with:")
    print("   - Episodic memory capacity: 1000")
    print("   - Working memory capacity: 10")
    print("   - Neural episodic control enabled")
    print("   - Long-term memory consolidation enabled")

    # Add experiences to memory
    print("\n💾 Adding experiences to memory...")

    for i in range(50):
        state = np.random.randn(64).astype(np.float32)
        action = np.random.randint(0, 4)
        reward = np.random.uniform(-1, 1)
        context = {
            "request": f"Task {i % 5}",  # Create patterns
            "episode": i,
            "difficulty": i / 50.0,
        }

        memory_system.add_experience(state, action, reward, context)

        if i % 10 == 0:
            print(f"   Added {i + 1} experiences...")

    # Test value estimation
    print("\n🔍 Testing value estimation...")

    test_state = np.random.randn(64).astype(np.float32)
    for action in range(4):
        value = memory_system.get_value_estimate(test_state, action)
        print(f"   Action {action}: Value = {value:.3f}")

    # Trigger memory consolidation
    print("\n🔄 Triggering memory consolidation...")
    memory_system.consolidate_memories()

    # Get memory statistics
    stats = memory_system.get_memory_statistics()
    print("\n📊 Memory Statistics:")
    print(f"   Episodic memories: {stats['episodic']['total_memories']}")
    print(f"   Working memory items: {stats['working_memory']['working_memory_items']}")
    print(f"   Consolidated memories: {stats['consolidated']['consolidated_memories']}")
    print(f"   Memory utilization: {stats['episodic']['memory_utilization']:.2f}")


async def demonstrate_few_shot_learning():
    """Demonstrate few-shot learning capabilities."""
    print("\n🎯 Demonstrating Few-Shot Learning")
    print("=" * 60)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    db = MemoryDatabase("few_shot_demo.db")
    reward_system = RewardSystem(db)

    # Create few-shot learning agent
    few_shot_agent = FewShotLearningAgent(
        name="few_shot_demo",
        model=model,
        db=db,
        reward_system=reward_system,
        state_dim=32,
        action_dim=3,
        k_shot=5,
    )

    print(f"✅ Created few-shot learning agent with k={few_shot_agent.k_shot}")

    # Add some experiences to memory
    print("\n💾 Adding experiences to episodic memory...")

    for i in range(20):
        state = np.random.randn(32).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.uniform(-1, 1)
        context = {
            "request": f"Pattern {i % 3}",
            "category": ["search", "analyze", "create"][i % 3],
        }

        few_shot_agent.add_to_memory(
            torch.FloatTensor(state), action, reward, context
        )

    # Test few-shot prediction
    print("\n🔮 Testing few-shot prediction...")

    test_state = np.random.randn(32).astype(np.float32)
    test_context = {
        "request": "Pattern 1",
        "category": "analyze",
    }

    action, confidence = few_shot_agent.few_shot_predict(
        torch.FloatTensor(test_state), test_context
    )

    print(f"   Predicted action: {action}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Memory size: {len(few_shot_agent.episodic_memory)}")


async def main():
    """Run all advanced RL demonstrations."""
    print("🚀 Advanced Reinforcement Learning Features Demonstration")
    print("=" * 80)

    try:
        # Import required libraries
        import torch
        globals()['torch'] = torch

        await demonstrate_meta_learning()
        await demonstrate_multi_agent_rl()
        await demonstrate_curriculum_learning()
        await demonstrate_advanced_memory()
        await demonstrate_few_shot_learning()

        print("\n🎉 All advanced RL demonstrations completed successfully!")
        print("\n📋 Summary of demonstrated features:")
        print("   ✅ Meta-Learning (MAML) - Fast adaptation to new tasks")
        print("   ✅ Multi-Agent RL - Cooperative and competitive learning")
        print("   ✅ Curriculum Learning - Progressive task difficulty")
        print("   ✅ Advanced Memory - Episodic, working, and consolidated memory")
        print("   ✅ Few-Shot Learning - Learning from minimal examples")

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
