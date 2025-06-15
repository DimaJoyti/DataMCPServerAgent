"""
Multi-agent learning entry point for DataMCPServerAgent.
This version implements a sophisticated multi-agent learning system with collaborative
problem-solving and knowledge sharing between agents.
"""

import asyncio
from typing import List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.agent_architecture import create_specialized_sub_agents
from src.agents.enhanced_agent_architecture import (
    EnhancedToolSelector,
    FeedbackCollector,
    LearningAgent,
    ToolPerformanceTracker,
)
from src.agents.multi_agent_learning import (
    CollaborativeLearningSystem,
    KnowledgeTransferAgent,
    create_multi_agent_learning_system,
)
from src.memory.collaborative_knowledge import create_collaborative_knowledge_base
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import BrightDataToolkit
from src.utils.agent_metrics import (
    create_agent_performance_tracker,
    create_multi_agent_performance_analyzer,
)

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

server_params = StdioServerParameters(
    model_name="claude-3-5-sonnet-20240620",
    model_provider="anthropic",
    user_id="user-123",
    conversation_id="conv-456",
)


async def load_all_tools(session: ClientSession) -> List[BaseTool]:
    """Load all available tools.

    Args:
        session: MCP client session

    Returns:
        List of tools
    """
    # Load MCP tools
    mcp_tools = await load_mcp_tools(session)

    # Load Bright Data tools
    bright_data_toolkit = BrightDataToolkit()
    bright_data_tools = bright_data_toolkit.get_tools()

    # Combine all tools
    all_tools = mcp_tools + bright_data_tools

    return all_tools


class MultiAgentLearningCoordinator:
    """Coordinator for multi-agent learning system."""

    def __init__(
        self, model: ChatAnthropic, tools: List[BaseTool], db_path: str = "multi_agent_memory.db"
    ):
        """Initialize the multi-agent learning coordinator.

        Args:
            model: Language model to use
            tools: List of available tools
            db_path: Path to the memory database
        """
        self.model = model
        self.tools = tools

        # Initialize memory database
        self.memory_db = MemoryDatabase(db_path)

        # Initialize collaborative knowledge base
        self.knowledge_base = create_collaborative_knowledge_base(self.memory_db)

        # Initialize performance tracking
        self.performance_tracker = create_agent_performance_tracker(self.memory_db)
        self.performance_analyzer = create_multi_agent_performance_analyzer(
            self.memory_db, self.performance_tracker
        )

        # Initialize tool performance tracker
        self.tool_tracker = ToolPerformanceTracker(self.memory_db)

        # Initialize enhanced tool selector
        self.tool_selector = EnhancedToolSelector(model, tools, self.memory_db, self.tool_tracker)

        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector(model, self.memory_db)

        # Create specialized sub-agents
        self.sub_agents = create_specialized_sub_agents(model, tools)

        # Create learning agents for each sub-agent
        self.learning_agents = {}
        for agent_name, agent in self.sub_agents.items():
            self.learning_agents[agent_name] = LearningAgent(
                agent.name, model, self.memory_db, self.feedback_collector
            )

        # Initialize knowledge transfer agent
        self.knowledge_transfer_agent = KnowledgeTransferAgent(model, self.memory_db)

        # Initialize collaborative learning system
        self.collaborative_learning = CollaborativeLearningSystem(
            model, self.memory_db, self.learning_agents, self.knowledge_transfer_agent
        )

        # Initialize multi-agent learning system
        self.multi_agent_learning = create_multi_agent_learning_system(
            model, self.memory_db, self.learning_agents, self.feedback_collector
        )

    async def process_request(self, request: str) -> str:
        """Process a user request using the multi-agent learning system.

        Args:
            request: User request

        Returns:
            Response to the user
        """
        # Select tools for the request
        tool_selection = await self.tool_selector.select_tools(request)

        # Determine which sub-agents to use based on selected tools
        selected_sub_agents = set()
        for tool_name in tool_selection["selected_tools"]:
            for agent_name, agent in self.sub_agents.items():
                if any(tool.name == tool_name for tool in agent.tools):
                    selected_sub_agents.add(agent_name)

        # If no sub-agents were selected, use the default agent
        if not selected_sub_agents and "default" in self.sub_agents:
            selected_sub_agents.add("default")

        # Execute selected sub-agents
        agent_results = {}
        for agent_name in selected_sub_agents:
            agent = self.sub_agents[agent_name]

            # Track performance
            start_time = time.time()

            # Execute the agent
            result = await agent.execute(request, self.memory_db)

            # Record performance
            execution_time = time.time() - start_time
            self.performance_tracker.record_agent_execution(
                agent_name, result["success"], execution_time
            )

            # Track tool performance
            for tool in agent.tools:
                if tool.name in tool_selection["selected_tools"]:
                    self.tool_tracker.end_execution(tool.name, result["success"])

            # Collect self-evaluation feedback
            if agent_name in self.learning_agents:
                await self.feedback_collector.perform_self_evaluation(
                    request,
                    result["response"] if result["success"] else result["error"],
                    agent_name,
                )

            agent_results[agent_name] = result

        # Process the request collaboratively
        collaborative_result = await self.multi_agent_learning.process_request_collaboratively(
            request, agent_results
        )

        # Record collaborative performance
        self.performance_tracker.record_collaborative_metric(
            "success_rate",
            1.0 if all(result["success"] for result in agent_results.values()) else 0.0,
            list(selected_sub_agents),
        )

        # Execute learning cycle periodically
        # This would typically be done asynchronously or on a schedule
        # For simplicity, we'll do it after every 10th request based on a counter in the database
        request_counter = self.memory_db.get_counter("request_counter")
        if request_counter % 10 == 0:
            await self.multi_agent_learning.execute_learning_cycle()

        # Increment request counter
        self.memory_db.increment_counter("request_counter")

        return collaborative_result["collaborative_solution"]


async def chat_with_multi_agent_learning_system():
    """Run the multi-agent learning system."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Initializing DataMCPServerAgent with multi-agent learning architecture...")

            # Load all tools
            tools = await load_all_tools(session)

            # Initialize multi-agent learning coordinator
            coordinator = MultiAgentLearningCoordinator(model, tools)

            # Process messages
            async for message in session.get_messages():
                if message.content:
                    try:
                        # Process the request
                        response = await coordinator.process_request(message.content)

                        # Send the response
                        await session.send_message(response)
                    except Exception as e:
                        error_message = f"Error processing request: {str(e)}"
                        await session.send_message(error_message)


if __name__ == "__main__":
    import time

    asyncio.run(chat_with_multi_agent_learning_system())
