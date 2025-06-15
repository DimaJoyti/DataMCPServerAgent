"""
Distributed memory entry point for DataMCPServerAgent.
This version implements a scalable distributed memory architecture.
"""

import asyncio
import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.agent_architecture import create_specialized_sub_agents
from src.agents.reinforcement_learning import (
    RewardSystem,
    RLCoordinatorAgent,
    create_rl_agent_architecture,
)
from src.memory.distributed_memory_manager import DistributedMemoryManager
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.error_handlers import format_error_for_user

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize distributed memory manager
memory_type = os.getenv("MEMORY_TYPE", "redis")
memory_config = {}

if memory_type == "redis":
    memory_config = {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB", "0")),
        "password": os.getenv("REDIS_PASSWORD", None),
        "prefix": f"{os.getenv('REDIS_PREFIX', 'datamcp')}:",
    }
elif memory_type == "mongodb":
    memory_config = {
        "connection_string": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
        "database_name": os.getenv("MONGODB_DB", "agent_memory"),
    }

memory_manager = DistributedMemoryManager(
    memory_type=memory_type, config=memory_config, namespace=os.getenv("MEMORY_NAMESPACE", "agent")
)


async def setup_rl_agent_with_distributed_memory(mcp_tools: List[BaseTool]) -> RLCoordinatorAgent:
    """Set up the reinforcement learning agent with distributed memory.

    Args:
        mcp_tools: List of MCP tools

    Returns:
        RL coordinator agent
    """
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mcp_tools)

    # Create reward system with distributed memory
    reward_system = RewardSystem(memory_manager)

    # Create RL coordinator agent
    rl_coordinator = await create_rl_agent_architecture(
        model=model,
        db=memory_manager,  # Use distributed memory manager instead of local DB
        sub_agents=sub_agents,
        rl_agent_type=os.getenv("RL_AGENT_TYPE", "q_learning"),
    )

    return rl_coordinator


async def chat_with_distributed_memory_agent() -> None:
    """Chat with the distributed memory agent."""
    # Check if the memory backend is accessible
    if not await memory_manager.backend.ping():
        logger.error(f"Cannot connect to {memory_type} backend. Please check your configuration.")
        print(f"Error: Cannot connect to {memory_type} backend. Please check your configuration.")
        return

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

            # Set up the RL agent with distributed memory
            rl_agent = await setup_rl_agent_with_distributed_memory(mcp_tools)

            print(f"\n=== Distributed Memory Agent ({memory_type.upper()}) ===\n")
            print(
                "Type 'exit' to quit, 'feedback: <message>' to provide feedback, 'learn' to perform batch learning, or 'memory' to view memory summary."
            )

            # Initialize conversation history
            history = []

            # Main chat loop
            while True:
                # Get user input
                user_input = input("\nYou: ")

                # Check for exit command
                if user_input.lower() == "exit":
                    break

                # Check for memory command
                if user_input.lower() == "memory":
                    try:
                        memory_summary = await memory_manager.get_memory_summary()
                        print("\n=== Memory Summary ===\n")
                        print(memory_summary)
                        continue
                    except Exception as e:
                        error_message = format_error_for_user(e)
                        print(f"\nError getting memory summary: {error_message}")
                        continue

                # Check for feedback command
                if user_input.lower().startswith("feedback:"):
                    if not history:
                        print("No conversation to provide feedback for.")
                        continue

                    feedback = user_input[len("feedback:") :].strip()

                    # Get the last request and response
                    last_request = next(
                        (msg["content"] for msg in reversed(history) if msg["role"] == "user"), None
                    )
                    last_response = next(
                        (msg["content"] for msg in reversed(history) if msg["role"] == "assistant"),
                        None,
                    )

                    if last_request and last_response:
                        # Update from feedback
                        await rl_agent.update_from_feedback(last_request, last_response, feedback)

                        # Save interaction for batch learning
                        await memory_manager.save_entity(
                            "agent_interaction",
                            f"interaction_{int(time.time())}",
                            {
                                "agent_name": "distributed_memory_agent",
                                "request": last_request,
                                "response": last_response,
                                "feedback": feedback,
                                "timestamp": time.time(),
                            },
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

                try:
                    # Process the request
                    result = await rl_agent.process_request(user_input, history)

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

                    # Save conversation history to distributed memory
                    await memory_manager.save_conversation_history(history)

                    # Print debug info
                    print(f"\n[Debug] Selected agent: {result['selected_agent']}")
                    print(f"[Debug] Reward: {result['reward']:.4f}")
                    print(f"[Debug] Memory backend: {memory_type}")

                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"\nError: {error_message}")


if __name__ == "__main__":
    asyncio.run(chat_with_distributed_memory_agent())
