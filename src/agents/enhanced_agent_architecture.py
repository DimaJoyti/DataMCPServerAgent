"""
Enhanced agent architecture for DataMCPServerAgent.
This module integrates memory persistence, enhanced tool selection, and learning capabilities.
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from src.agents.agent_architecture import SpecializedSubAgent, create_specialized_sub_agents
from src.tools.enhanced_tool_selection import EnhancedToolSelector, ToolPerformanceTracker
from src.utils.error_handlers import format_error_for_user
from src.agents.learning_capabilities import FeedbackCollector, LearningAgent
from src.memory.memory_persistence import MemoryDatabase


class EnhancedCoordinatorAgent:
    """Enhanced coordinator agent with learning capabilities."""
    
    def __init__(
        self, 
        model: ChatAnthropic, 
        sub_agents: Dict[str, SpecializedSubAgent],
        tool_selector: EnhancedToolSelector,
        memory_db: MemoryDatabase,
        performance_tracker: ToolPerformanceTracker,
        feedback_collector: FeedbackCollector,
        learning_agents: Dict[str, LearningAgent]
    ):
        """Initialize the enhanced coordinator agent.
        
        Args:
            model: Language model to use
            sub_agents: Dictionary of sub-agents by name
            tool_selector: Enhanced tool selection agent
            memory_db: Memory database for persistence
            performance_tracker: Tool performance tracker
            feedback_collector: Feedback collector
            learning_agents: Dictionary of learning agents by name
        """
        self.model = model
        self.sub_agents = sub_agents
        self.tool_selector = tool_selector
        self.memory_db = memory_db
        self.performance_tracker = performance_tracker
        self.feedback_collector = feedback_collector
        self.learning_agents = learning_agents
        
        # Load conversation history from the database
        self.conversation_history = self.memory_db.load_conversation_history()
        
        # Create the coordinator prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an enhanced coordinator agent responsible for managing multiple specialized sub-agents to complete complex tasks.
Your job is to:
1. Analyze the user's request in detail
2. Break it down into subtasks
3. Assign each subtask to the appropriate sub-agent
4. Monitor the execution of each sub-agent
5. Synthesize the results into a coherent response
6. Learn from the interaction to improve future performance

Available sub-agents:
{sub_agent_descriptions}

When responding, first explain your plan for completing the task, then show the results from each sub-agent, and finally provide a synthesized answer.
"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{request}")
        ])
    
    async def process_request(self, request: str) -> str:
        """Process a user request by coordinating sub-agents.
        
        Args:
            request: User request
            
        Returns:
            Response to the user
        """
        # Add the user request to conversation history
        self.conversation_history.append({"role": "user", "content": request})
        
        # Get recent conversation history
        history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
        
        # Select tools for the request
        tool_selection = await self.tool_selector.select_tools(request, history)
        
        # Determine which sub-agents to use based on selected tools
        selected_sub_agents = set()
        for tool_name in tool_selection["selected_tools"]:
            for agent_name, agent in self.sub_agents.items():
                if any(tool.name == tool_name for tool in agent.tools):
                    selected_sub_agents.add(agent_name)
        
        # If no sub-agents were selected, use the default agent
        if not selected_sub_agents and "default" in self.sub_agents:
            selected_sub_agents.add("default")
        
        # Format sub-agent descriptions
        sub_agent_descriptions = "\n".join([
            f"- {name}: {agent.name}" for name, agent in self.sub_agents.items()
            if name in selected_sub_agents
        ])
        
        # Prepare the input for the coordinator prompt
        input_values = {
            "request": request,
            "sub_agent_descriptions": sub_agent_descriptions,
            "history": history
        }
        
        # Get the coordination plan
        messages = self.prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)
        plan = response.content
        
        # Execute the plan using selected sub-agents
        results = []
        for agent_name in selected_sub_agents:
            agent = self.sub_agents[agent_name]
            
            # Track tool performance for each tool used by this agent
            for tool in agent.tools:
                self.performance_tracker.start_execution(tool.name)
            
            # Execute the agent
            result = await agent.execute(request, self.memory_db)
            
            # Record tool performance
            for tool in agent.tools:
                self.performance_tracker.end_execution(tool.name, result["success"])
            
            # Collect self-evaluation feedback
            if agent_name in self.learning_agents:
                await self.feedback_collector.perform_self_evaluation(
                    request,
                    result["response"] if result["success"] else result["error"],
                    agent_name
                )
            
            results.append(result)
        
        # Synthesize the results
        synthesis_prompt = f"""
You have received results from multiple sub-agents for the following request:
{request}

Sub-agent results:
{json.dumps(results, indent=2)}

Based on these results, provide a comprehensive response to the user's request.
Incorporate information from all sub-agents and resolve any conflicts or inconsistencies.
"""
        
        synthesis_messages = [
            {"role": "system", "content": "You are a synthesis agent that combines results from multiple sub-agents into a coherent response."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        synthesis_response = await self.model.ainvoke(synthesis_messages)
        final_response = synthesis_response.content
        
        # Add the final response to conversation history
        self.conversation_history.append({"role": "assistant", "content": final_response})
        
        # Save conversation history to the database
        self.memory_db.save_conversation_history(self.conversation_history)
        
        # Provide tool execution feedback
        for tool_name in tool_selection["selected_tools"]:
            await self.tool_selector.provide_execution_feedback(
                request,
                tool_name,
                {"request": request},  # Simplified args
                "Success" if any(r["success"] for r in results) else "Failed",
                1.0,  # Simplified execution time
                any(r["success"] for r in results)
            )
        
        return final_response
    
    async def collect_user_feedback(self, request: str, response: str, feedback: str) -> None:
        """Collect user feedback on a response.
        
        Args:
            request: Original user request
            response: Agent response
            feedback: User feedback
        """
        # Collect feedback for each learning agent
        for agent_name, learning_agent in self.learning_agents.items():
            await self.feedback_collector.collect_user_feedback(
                request,
                response,
                feedback,
                agent_name
            )
    
    async def learn_from_feedback(self) -> Dict[str, str]:
        """Learn from collected feedback to improve future performance.
        
        Returns:
            Dictionary of learning insights by agent name
        """
        insights = {}
        
        # Learn from feedback for each learning agent
        for agent_name, learning_agent in self.learning_agents.items():
            agent_insights = await learning_agent.learn_from_feedback()
            insights[agent_name] = agent_insights
        
        return insights
    
    async def get_learning_insights(self) -> str:
        """Get a summary of learning insights from all agents.
        
        Returns:
            Summary of learning insights
        """
        insights_summary = "# Learning Insights Summary\n\n"
        
        # Get insights from each learning agent
        for agent_name, learning_agent in self.learning_agents.items():
            agent_insights = await learning_agent.get_learning_insights()
            insights_summary += f"## {agent_name}\n\n{agent_insights}\n\n"
        
        return insights_summary


# Factory function to create enhanced agent architecture
async def create_enhanced_agent_architecture(
    model: ChatAnthropic,
    tools: List[BaseTool],
    db_path: str = "agent_memory.db"
) -> EnhancedCoordinatorAgent:
    """Create an enhanced agent architecture with memory persistence, tool selection, and learning.
    
    Args:
        model: Language model to use
        tools: List of available tools
        db_path: Path to the memory database
        
    Returns:
        Enhanced coordinator agent
    """
    # Initialize memory database
    memory_db = MemoryDatabase(db_path)
    
    # Initialize tool performance tracker
    performance_tracker = ToolPerformanceTracker(memory_db)
    
    # Initialize enhanced tool selector
    tool_selector = EnhancedToolSelector(model, tools, memory_db, performance_tracker)
    
    # Initialize feedback collector
    feedback_collector = FeedbackCollector(model, memory_db)
    
    # Create specialized sub-agents
    sub_agents = create_specialized_sub_agents(model, tools)
    
    # Create learning agents for each sub-agent
    learning_agents = {}
    for agent_name, agent in sub_agents.items():
        learning_agents[agent_name] = LearningAgent(agent.name, model, memory_db, feedback_collector)
    
    # Create enhanced coordinator agent
    coordinator = EnhancedCoordinatorAgent(
        model,
        sub_agents,
        tool_selector,
        memory_db,
        performance_tracker,
        feedback_collector,
        learning_agents
    )
    
    return coordinator