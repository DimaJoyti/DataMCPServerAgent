"""
Example script demonstrating the reinforcement learning capabilities.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.agents.agent_architecture import create_specialized_sub_agents
from src.agents.reinforcement_learning import (
    create_rl_agent_architecture,
)
from src.memory.memory_persistence import MemoryDatabase

load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize memory database
db = MemoryDatabase("reinforcement_learning_example.db")

# Sample requests for testing
SAMPLE_REQUESTS = [
    "Search for information about reinforcement learning in AI agents",
    "Find the latest news about artificial intelligence",
    "Compare prices of iPhone 14 and iPhone 15",
    "Analyze the sentiment of tweets about climate change",
    "Scrape product information from Amazon for gaming laptops",
    "Find reviews of the latest Samsung Galaxy phone",
    "Get information about the weather in New York",
    "Find the best restaurants in San Francisco",
    "Analyze the performance of tech stocks in the last month",
    "Find the most popular Python libraries for machine learning"
]

# Sample feedback for testing
SAMPLE_FEEDBACK = [
    "That was very helpful, thank you!",
    "Good job finding that information.",
    "This is exactly what I was looking for.",
    "The analysis could be more detailed.",
    "You missed some important information.",
    "Great comparison, very thorough!",
    "The sentiment analysis seems inaccurate.",
    "Perfect, this saves me a lot of time.",
    "I needed more specific information.",
    "Excellent work, very comprehensive!"
]

async def simulate_rl_agent_learning(rl_agent_type: str = "q_learning") -> None:
    """Simulate learning with a reinforcement learning agent.

    Args:
        rl_agent_type: Type of RL agent to use ("q_learning" or "policy_gradient")
    """
    print(f"\n=== Simulating {rl_agent_type.upper()} Agent Learning ===\n")

    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, [])

    # Create RL coordinator agent
    rl_coordinator = await create_rl_agent_architecture(
        model=model,
        db=db,
        sub_agents=sub_agents,
        rl_agent_type=rl_agent_type
    )

    # Simulate interactions
    history = []

    for i, request in enumerate(SAMPLE_REQUESTS):
        print(f"\nRequest {i+1}: {request}")

        # Process the request
        result = await rl_coordinator.process_request(request, history)

        # Print the result
        if result["success"]:
            print(f"Response: {result['response'][:100]}...")
        else:
            print(f"Error: {result['error']}")

        print(f"Selected agent: {result['selected_agent']}")
        print(f"Reward: {result['reward']:.4f}")

        # Add to history
        history.append({"role": "user", "content": request})
        history.append({
            "role": "assistant",
            "content": result["response"] if result["success"] else result["error"]
        })

        # Provide feedback (simulated)
        feedback = SAMPLE_FEEDBACK[i]
        print(f"Feedback: {feedback}")

        # Update from feedback
        await rl_coordinator.update_from_feedback(
            request,
            result["response"] if result["success"] else result["error"],
            feedback
        )

        # Save interaction for batch learning
        db.save_agent_interaction(
            "rl_coordinator",
            request,
            result["response"] if result["success"] else result["error"],
            feedback
        )

        # Wait a bit
        await asyncio.sleep(1)

    # Perform batch learning
    print("\n=== Performing Batch Learning ===\n")
    learning_result = await rl_coordinator.learn_from_batch()
    print(f"Learning result: {learning_result}")

    # Print memory summary
    print("\n=== Memory Summary ===\n")
    print(db.get_memory_summary())

async def compare_rl_approaches() -> None:
    """Compare different reinforcement learning approaches."""
    print("\n=== Comparing Reinforcement Learning Approaches ===\n")

    # Simulate Q-learning
    await simulate_rl_agent_learning("q_learning")

    # Simulate policy gradient
    await simulate_rl_agent_learning("policy_gradient")

    print("\n=== Comparison Complete ===\n")
    print("Both approaches have been tested and their results are stored in the database.")
    print("Q-learning is better for discrete state spaces and simpler problems.")
    print("Policy gradient is better for continuous state spaces and more complex problems.")

async def main() -> None:
    """Main function."""
    # Compare RL approaches
    await compare_rl_approaches()

if __name__ == "__main__":
    asyncio.run(main())
