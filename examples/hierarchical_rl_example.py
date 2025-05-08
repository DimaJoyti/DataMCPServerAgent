"""
Example script demonstrating the hierarchical reinforcement learning capabilities.
"""

import asyncio
import os
import sys
import time
import uuid
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from src.agents.agent_architecture import (
    SpecializedSubAgent,
    create_specialized_sub_agents
)
from src.agents.hierarchical_rl import (
    HierarchicalRLCoordinatorAgent,
    HierarchicalRewardSystem,
    Option,
    create_hierarchical_rl_agent_architecture
)
from src.memory.hierarchical_memory_persistence import HierarchicalMemoryDatabase
from src.utils.error_handlers import format_error_for_user

load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize hierarchical memory database
db = HierarchicalMemoryDatabase("hierarchical_rl_example.db")


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


async def create_mock_sub_agents(model: ChatAnthropic, tools: List[BaseTool]) -> Dict[str, SpecializedSubAgent]:
    """Create mock sub-agents for testing.
    
    Args:
        model: Language model to use
        tools: List of tools
        
    Returns:
        Dictionary of sub-agents
    """
    # Create specialized sub-agents
    sub_agents = {}
    
    # Search agent
    search_agent = SpecializedSubAgent(
        name="search_agent",
        description="Agent specialized in searching for information",
        model=model,
        tools=[tools[0]],  # search_tool
    )
    sub_agents["search_agent"] = search_agent
    
    # Calculator agent
    calculator_agent = SpecializedSubAgent(
        name="calculator_agent",
        description="Agent specialized in performing calculations",
        model=model,
        tools=[tools[1]],  # calculator_tool
    )
    sub_agents["calculator_agent"] = calculator_agent
    
    # Weather agent
    weather_agent = SpecializedSubAgent(
        name="weather_agent",
        description="Agent specialized in providing weather information",
        model=model,
        tools=[tools[2]],  # weather_tool
    )
    sub_agents["weather_agent"] = weather_agent
    
    # Translation agent
    translation_agent = SpecializedSubAgent(
        name="translation_agent",
        description="Agent specialized in translating text",
        model=model,
        tools=[tools[3]],  # translation_tool
    )
    sub_agents["translation_agent"] = translation_agent
    
    # Summarization agent
    summarization_agent = SpecializedSubAgent(
        name="summarization_agent",
        description="Agent specialized in summarizing text",
        model=model,
        tools=[tools[4]],  # summarization_tool
    )
    sub_agents["summarization_agent"] = summarization_agent
    
    return sub_agents


async def create_options(
    agent_name: str,
    sub_agents: Dict[str, SpecializedSubAgent],
    db: HierarchicalMemoryDatabase
) -> List[Option]:
    """Create options for hierarchical reinforcement learning.
    
    Args:
        agent_name: Name of the agent
        sub_agents: Dictionary of sub-agents
        db: Hierarchical memory database
        
    Returns:
        List of options
    """
    options = []
    
    # Create an option for each sub-agent
    for sub_agent_name, sub_agent in sub_agents.items():
        # Define initiation states based on the sub-agent's specialty
        if sub_agent_name == "search_agent":
            initiation_states = ["search_query", "information_request", "question"]
        elif sub_agent_name == "calculator_agent":
            initiation_states = ["calculation_request", "math_query"]
        elif sub_agent_name == "weather_agent":
            initiation_states = ["weather_query", "forecast_request"]
        elif sub_agent_name == "translation_agent":
            initiation_states = ["translation_request", "language_query"]
        elif sub_agent_name == "summarization_agent":
            initiation_states = ["summarization_request", "summary_query"]
        else:
            initiation_states = []
        
        # Define termination states
        termination_states = ["task_completed", "error_state"]
        
        # Define policy mapping
        policy_mapping = {
            state: sub_agent_name for state in initiation_states
        }
        
        # Create the option
        option = Option.create_option(
            option_name=f"Use {sub_agent_name}",
            initiation_states=initiation_states,
            termination_states=termination_states,
            policy_mapping=policy_mapping,
            db=db,
            agent_name=agent_name,
        )
        
        options.append(option)
    
    # Create a composite option for complex tasks
    composite_option = Option.create_option(
        option_name="Complex task handler",
        initiation_states=["complex_query", "multi_step_task"],
        termination_states=["task_completed", "error_state"],
        policy_mapping={
            "complex_query": "search_agent",
            "multi_step_task": "search_agent",
            "calculation_needed": "calculator_agent",
            "translation_needed": "translation_agent",
            "summarization_needed": "summarization_agent",
        },
        db=db,
        agent_name=agent_name,
    )
    
    options.append(composite_option)
    
    return options


async def demonstrate_hierarchical_rl() -> None:
    """Demonstrate hierarchical reinforcement learning."""
    print("\n=== Demonstrating Hierarchical Reinforcement Learning ===\n")
    
    # Create mock tools
    mock_tools = await create_mock_tools()
    
    # Create mock sub-agents
    sub_agents = await create_mock_sub_agents(model, mock_tools)
    
    # Create hierarchical reward system
    reward_system = HierarchicalRewardSystem(db)
    
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
    
    # Create state extraction prompt
    state_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a state extraction agent responsible for converting user requests into state representations.
Your job is to analyze a user request and extract key features that can be used to determine the appropriate agent to handle it.

For each request, you should:
1. Identify the main task type (search, calculation, weather, translation, summarization, etc.)
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
    
    # Create options
    options = await create_options("hierarchical_rl_demo", sub_agents, db)
    
    # Create hierarchical RL coordinator agent
    hierarchical_rl_coordinator = HierarchicalRLCoordinatorAgent(
        name="hierarchical_rl_demo",
        model=model,
        db=db,
        reward_system=reward_system,
        sub_agents=sub_agents,
        tools=mock_tools,
        task_decomposition_prompt=task_decomposition_prompt,
    )
    
    # Process a few requests
    requests = [
        "What is the weather in New York?",
        "Calculate 15 * 7",
        "Translate 'hello' to Spanish",
        "Search for information about hierarchical reinforcement learning",
        "Summarize this article about climate change",
        "I need to plan a trip to Paris. Find the weather, translate some basic phrases, and summarize the top attractions.",
    ]
    
    for request in requests:
        print(f"\nProcessing request: {request}")
        
        # Extract state
        context = {"request": request, "history": []}
        state = await hierarchical_rl_coordinator._extract_state(context)
        print(f"Extracted state: {state}")
        
        # Decompose task
        task_decomposition = await hierarchical_rl_coordinator._decompose_task(request, [])
        print(f"Task decomposition:")
        print(f"- Task ID: {task_decomposition['task_id']}")
        print(f"- Task name: {task_decomposition['task_name']}")
        print(f"- Subtasks:")
        for subtask in task_decomposition['subtasks']:
            print(f"  - {subtask['name']}: {subtask['description']}")
        
        # Process request
        result = await hierarchical_rl_coordinator.process_request(request, [])
        print(f"Selected option: {result['selected_option']}")
        print(f"Reward: {result['reward']:.4f}")
        
        # Add a separator
        print("\n" + "-" * 50)
    
    # Get subtask history
    subtask_history = db.get_subtask_history("hierarchical_rl_demo", limit=10)
    
    print("\nSubtask History:")
    for subtask in subtask_history:
        print(f"- {subtask['subtask_name']} (Task: {subtask['task_id']})")
        print(f"  - State: {subtask['state']}")
        print(f"  - Action: {subtask['action']}")
        print(f"  - Reward: {subtask['reward']:.4f}")
        print(f"  - Success: {subtask['success']}")
        print(f"  - Duration: {subtask['duration']:.2f}s")
        print()


async def main() -> None:
    """Main function."""
    print("=== Hierarchical Reinforcement Learning Example ===")
    
    # Demonstrate hierarchical RL
    await demonstrate_hierarchical_rl()
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
