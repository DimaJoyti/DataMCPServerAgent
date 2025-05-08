"""
Reinforcement learning entry point for DataMCPServerAgent.
This version implements a sophisticated reinforcement learning system for continuous improvement.
"""

import asyncio
import os
from typing import Dict, List, Any

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
    create_rl_agent_architecture
)
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.error_handlers import format_error_for_user

load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize memory database
db_path = os.getenv("MEMORY_DB_PATH", "agent_memory.db")
db = MemoryDatabase(db_path)


async def setup_rl_agent(mcp_tools: List[BaseTool]) -> RLCoordinatorAgent:
    """Set up the reinforcement learning agent.
    
    Args:
        mcp_tools: List of MCP tools
        
    Returns:
        RL coordinator agent
    """
    # Create specialized sub-agents
    sub_agents = await create_specialized_sub_agents(model, mcp_tools)
    
    # Create RL coordinator agent
    rl_coordinator = await create_rl_agent_architecture(
        model=model,
        db=db,
        sub_agents=sub_agents,
        rl_agent_type=os.getenv("RL_AGENT_TYPE", "q_learning")
    )
    
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
            rl_agent = await setup_rl_agent(mcp_tools)
            
            print("\n=== Reinforcement Learning Agent ===\n")
            print("Type 'exit' to quit, 'feedback: <message>' to provide feedback, or 'learn' to perform batch learning.")
            
            # Initialize conversation history
            history = []
            
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
                    
                    feedback = user_input[len("feedback:"):].strip()
                    
                    # Get the last request and response
                    last_request = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), None)
                    last_response = next((msg["content"] for msg in reversed(history) if msg["role"] == "assistant"), None)
                    
                    if last_request and last_response:
                        # Update from feedback
                        await rl_agent.update_from_feedback(last_request, last_response, feedback)
                        
                        # Save interaction for batch learning
                        db.save_agent_interaction("rl_coordinator", last_request, last_response, feedback)
                        
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
                    history.append({
                        "role": "assistant", 
                        "content": result["response"] if result["success"] else result["error"]
                    })
                    
                    # Print debug info
                    print(f"\n[Debug] Selected agent: {result['selected_agent']}")
                    print(f"[Debug] Reward: {result['reward']:.4f}")
                    
                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"\nError: {error_message}")


if __name__ == "__main__":
    asyncio.run(chat_with_rl_agent())
