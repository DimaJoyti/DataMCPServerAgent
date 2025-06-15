"""
Enhanced reinforcement learning entry point for DataMCPServerAgent.
This version implements modern deep RL algorithms with advanced state representation.
"""

import asyncio
import os
from typing import List, Union

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.advanced_rl_techniques import RainbowDQNAgent
from src.agents.agent_architecture import create_specialized_sub_agents
from src.agents.enhanced_state_representation import (
    ContextualStateEncoder,
    GraphStateEncoder,
    TextEmbeddingEncoder,
)
from src.agents.modern_deep_rl import (
    ModernDeepRLCoordinatorAgent,
    create_modern_deep_rl_agent_architecture,
)
from src.memory.advanced_memory_persistence import (
    AdvancedMemoryDatabase as MemoryDatabase,
)

# Load environment variables
load_dotenv()

# Initialize components
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4000,
)

# Initialize database
db = MemoryDatabase("enhanced_rl_agent_memory.db")


async def setup_enhanced_rl_agent(
    mcp_tools: List[BaseTool],
    rl_algorithm: str = "auto",
    state_representation: str = "contextual"
) -> Union[ModernDeepRLCoordinatorAgent, RainbowDQNAgent]:
    """Set up the enhanced reinforcement learning agent.

    Args:
        mcp_tools: List of MCP tools
        rl_algorithm: RL algorithm to use ("dqn", "ppo", "a2c", "rainbow", "auto")
        state_representation: State representation type ("simple", "contextual", "graph")

    Returns:
        Enhanced RL coordinator agent
    """
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mcp_tools)

    # Determine RL algorithm if auto
    if rl_algorithm == "auto":
        rl_algorithm = os.getenv("RL_ALGORITHM", "dqn")

    # Create state encoder based on type
    if state_representation == "contextual":
        text_encoder = TextEmbeddingEncoder()
        state_encoder = ContextualStateEncoder(
            text_encoder=text_encoder,
            include_temporal=True,
            include_performance=True,
            include_user_profile=True,
        )
    elif state_representation == "graph":
        state_encoder = GraphStateEncoder(embedding_dim=256)
    else:
        state_encoder = None  # Use simple state representation

    # Create enhanced RL coordinator agent
    if rl_algorithm == "rainbow":
        # Special handling for Rainbow DQN
        from src.agents.reinforcement_learning import RewardSystem
        reward_system = RewardSystem(db)

        # For Rainbow DQN, we need to handle state encoding differently
        coordinator = RainbowDQNAgent(
            name="rainbow_dqn_coordinator",
            model=model,
            db=db,
            reward_system=reward_system,
            state_dim=512,  # Fixed state dimension
            action_dim=len(sub_agents) + len(mcp_tools),
        )
    else:
        # Use modern deep RL coordinator
        coordinator = await create_modern_deep_rl_agent_architecture(
            model=model,
            db=db,
            sub_agents=sub_agents,
            tools=mcp_tools,
            rl_algorithm=rl_algorithm,
            state_encoder=state_encoder,
        )

    return coordinator


async def chat_with_enhanced_rl_agent():
    """Main chat loop with enhanced RL agent."""
    print("🤖 Enhanced RL DataMCPServerAgent starting up...")
    print("🧠 Loading modern deep RL algorithms...")

    # Load MCP tools
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@brightdata/mcp-server-bright-data"],
        env={"BRIGHT_DATA_API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN")},
    )

    mcp_tools = []
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = load_mcp_tools(session)
                print(f"✅ Loaded {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"⚠️ Could not load MCP tools: {e}")
        print("🔄 Continuing with basic functionality...")

    # Get RL configuration from environment
    rl_algorithm = os.getenv("RL_ALGORITHM", "dqn")
    state_representation = os.getenv("STATE_REPRESENTATION", "contextual")

    print(f"🎯 Using RL algorithm: {rl_algorithm}")
    print(f"🧠 Using state representation: {state_representation}")

    # Set up enhanced RL agent
    enhanced_rl_agent = await setup_enhanced_rl_agent(
        mcp_tools, rl_algorithm, state_representation
    )

    print("✅ Enhanced RL agent ready!")
    print("💡 Features enabled:")
    print("   - Modern deep RL algorithms (DQN, PPO, A2C, Rainbow)")
    print("   - Advanced state representation")
    print("   - Prioritized experience replay")
    print("   - Multi-step learning")
    print("   - Noisy networks for exploration")
    print("   - Dueling architecture")
    print("   - Distributional RL (Rainbow)")
    print("\n🎮 Type 'quit' to exit, 'help' for commands")

    conversation_history = []
    episode_count = 0

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'help':
                print("\n📋 Available commands:")
                print("  help - Show this help message")
                print("  stats - Show training statistics")
                print("  save - Save model weights")
                print("  train - Force training step")
                print("  reset - Reset conversation history")
                print("  quit - Exit the agent")
                continue

            if user_input.lower() == 'stats':
                print("\n📊 Training Statistics:")
                print(f"   Episodes: {episode_count}")
                if hasattr(enhanced_rl_agent, 'rl_agent'):
                    if hasattr(enhanced_rl_agent.rl_agent, 'steps'):
                        print(f"   Training steps: {enhanced_rl_agent.rl_agent.steps}")
                    if hasattr(enhanced_rl_agent.rl_agent, 'epsilon'):
                        print(f"   Exploration rate: {enhanced_rl_agent.rl_agent.epsilon:.3f}")
                continue

            if user_input.lower() == 'save':
                try:
                    if hasattr(enhanced_rl_agent, 'rl_agent') and hasattr(enhanced_rl_agent.rl_agent, 'save_model'):
                        enhanced_rl_agent.rl_agent.save_model(f"enhanced_rl_model_{rl_algorithm}.pth")
                        print("💾 Model saved successfully!")
                    else:
                        print("⚠️ Model saving not supported for this agent type")
                except Exception as e:
                    print(f"❌ Error saving model: {e}")
                continue

            if user_input.lower() == 'train':
                try:
                    if hasattr(enhanced_rl_agent, 'train_episode'):
                        metrics = await enhanced_rl_agent.train_episode()
                        print(f"🎯 Training metrics: {metrics}")
                    else:
                        print("⚠️ Training not available for this agent type")
                except Exception as e:
                    print(f"❌ Error during training: {e}")
                continue

            if user_input.lower() == 'reset':
                conversation_history.clear()
                episode_count = 0
                print("🔄 Conversation history reset!")
                continue

            if not user_input:
                continue

            print("🤔 Processing with enhanced RL...")

            # Process request with enhanced RL agent
            if hasattr(enhanced_rl_agent, 'process_request'):
                result = await enhanced_rl_agent.process_request(user_input, conversation_history)
            else:
                # Handle Rainbow DQN differently
                result = {
                    "success": True,
                    "response": "Enhanced RL processing (Rainbow DQN implementation in progress)",
                    "selected_action": "default",
                    "reward": 0.5,
                }

            # Display result
            print(f"\n🤖 Agent: {result.get('response', 'No response')}")

            if result.get('selected_action'):
                print(f"🎯 Selected action: {result['selected_action']}")

            if 'reward' in result:
                print(f"🏆 Reward: {result['reward']:.3f}")

            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            conversation_history.append({
                "role": "assistant",
                "content": result.get('response', '')
            })

            # Keep history manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

            # Train the agent
            if hasattr(enhanced_rl_agent, 'train_episode'):
                try:
                    training_metrics = await enhanced_rl_agent.train_episode()
                    if training_metrics:
                        print(f"📈 Training: {training_metrics}")
                except Exception as e:
                    print(f"⚠️ Training error: {e}")

            episode_count += 1

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Continuing...")


if __name__ == "__main__":
    asyncio.run(chat_with_enhanced_rl_agent())
