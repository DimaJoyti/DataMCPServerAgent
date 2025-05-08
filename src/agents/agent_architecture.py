"""
Advanced agent architecture for DataMCPServerAgent.
This module implements specialized sub-agents, tool selection, and agent coordination.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from src.utils.error_handlers import format_error_for_user


class AgentMemory:
    """Memory system for storing conversation history and agent state."""
    
    def __init__(self, max_history_length: int = 20):
        """Initialize the agent memory.
        
        Args:
            max_history_length: Maximum number of messages to keep in history
        """
        self.conversation_history = []
        self.tool_usage_history = {}
        self.entity_memory = {}
        self.max_history_length = max_history_length
    
    def add_message(self, message: Dict[str, str]) -> None:
        """Add a message to the conversation history.
        
        Args:
            message: Message to add (dict with 'role' and 'content' keys)
        """
        self.conversation_history.append(message)
        
        # Trim history if it exceeds the maximum length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def add_tool_usage(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Record a tool usage in the history.
        
        Args:
            tool_name: Name of the tool used
            args: Arguments passed to the tool
            result: Result returned by the tool
        """
        if tool_name not in self.tool_usage_history:
            self.tool_usage_history[tool_name] = []
        
        self.tool_usage_history[tool_name].append({
            "args": args,
            "result": result,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def add_entity(self, entity_type: str, entity_id: str, data: Dict[str, Any]) -> None:
        """Add or update an entity in memory.
        
        Args:
            entity_type: Type of entity (e.g., 'product', 'website', 'person')
            entity_id: Unique identifier for the entity
            data: Entity data
        """
        if entity_type not in self.entity_memory:
            self.entity_memory[entity_type] = {}
        
        self.entity_memory[entity_type][entity_id] = {
            **data,
            "last_updated": asyncio.get_event_loop().time()
        }
    
    def get_recent_messages(self, n: int = 5) -> List[Dict[str, str]]:
        """Get the n most recent messages.
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            List of recent messages
        """
        return self.conversation_history[-n:]
    
    def get_tool_usage(self, tool_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get tool usage history.
        
        Args:
            tool_name: Name of tool to get history for, or None for all tools
            
        Returns:
            Tool usage history
        """
        if tool_name:
            return {tool_name: self.tool_usage_history.get(tool_name, [])}
        return self.tool_usage_history
    
    def get_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get an entity from memory.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            
        Returns:
            Entity data or None if not found
        """
        return self.entity_memory.get(entity_type, {}).get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            Dictionary of entities by ID
        """
        return self.entity_memory.get(entity_type, {})
    
    def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.
        
        Returns:
            Summary string
        """
        summary = "## Memory Summary\n\n"
        
        # Conversation summary
        summary += f"### Conversation History\n"
        summary += f"- {len(self.conversation_history)} messages in history\n"
        
        # Tool usage summary
        summary += f"\n### Tool Usage\n"
        for tool_name, usages in self.tool_usage_history.items():
            summary += f"- {tool_name}: {len(usages)} uses\n"
        
        # Entity memory summary
        summary += f"\n### Entities in Memory\n"
        for entity_type, entities in self.entity_memory.items():
            summary += f"- {entity_type}: {len(entities)} entities\n"
        
        return summary


class ToolSelectionAgent:
    """Agent responsible for selecting the most appropriate tools for a task."""
    
    def __init__(self, model: ChatAnthropic, tools: List[BaseTool]):
        """Initialize the tool selection agent.
        
        Args:
            model: Language model to use
            tools: List of available tools
        """
        self.model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        
        # Create the tool selection prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a specialized agent responsible for selecting the most appropriate tools for a given task.
Your job is to analyze the user's request and determine which tools would be most effective for completing it.

For each request, you should:
1. Analyze the task requirements
2. Consider the available tools and their capabilities
3. Select 1-3 tools that would be most appropriate for the task
4. Explain your reasoning for each tool selection

Respond with a JSON object containing:
- "selected_tools": Array of tool names
- "reasoning": Brief explanation of your selection
- "execution_order": Suggested order to use the tools (array of tool names)

Be strategic in your selection - choose tools that complement each other and cover all aspects of the task.
"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="""
User request: {request}

Available tools:
{tool_descriptions}

Select the most appropriate tools for this task.
""")
        ])
    
    async def select_tools(self, request: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Select the most appropriate tools for a request.
        
        Args:
            request: User request
            history: Optional conversation history
            
        Returns:
            Dictionary with selected tools, reasoning, and execution order
        """
        # Format tool descriptions
        tool_descriptions = "\n\n".join([
            f"- {tool.name}: {tool.description}" for tool in self.tools
        ])
        
        # Prepare the input for the prompt
        input_values = {
            "request": request,
            "tool_descriptions": tool_descriptions,
            "history": history or []
        }
        
        # Get the tool selection from the model
        messages = self.prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)
        
        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            json_str = content.split("```json")[1].split("```")[0] if "```json" in content else content
            json_str = json_str.strip()
            
            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]
            
            result = json.loads(json_str)
            
            # Validate the result
            if "selected_tools" not in result:
                result["selected_tools"] = []
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided."
            if "execution_order" not in result:
                result["execution_order"] = result["selected_tools"]
            
            # Filter out any tools that don't exist
            result["selected_tools"] = [t for t in result["selected_tools"] if t in self.tool_map]
            result["execution_order"] = [t for t in result["execution_order"] if t in self.tool_map]
            
            return result
        except Exception as e:
            # If parsing fails, return a default selection
            return {
                "selected_tools": [self.tools[0].name] if self.tools else [],
                "reasoning": f"Error parsing tool selection: {str(e)}. Defaulting to first available tool.",
                "execution_order": [self.tools[0].name] if self.tools else []
            }


class SpecializedSubAgent:
    """Base class for specialized sub-agents that focus on specific tasks."""
    
    def __init__(self, name: str, model: ChatAnthropic, tools: List[BaseTool], system_prompt: str):
        """Initialize the specialized sub-agent.
        
        Args:
            name: Name of the sub-agent
            model: Language model to use
            tools: List of tools available to this sub-agent
            system_prompt: System prompt for the sub-agent
        """
        self.name = name
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        
        # Create the ReAct agent
        self.agent = create_react_agent(model, tools, system_prompt)
    
    async def execute(self, task: str, memory: AgentMemory) -> Dict[str, Any]:
        """Execute a task using this sub-agent.
        
        Args:
            task: Task description
            memory: Agent memory
            
        Returns:
            Execution result
        """
        # Prepare the messages with memory context
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add relevant context from memory
        recent_messages = memory.get_recent_messages(3)
        messages.extend(recent_messages)
        
        # Add the current task
        messages.append({"role": "user", "content": task})
        
        try:
            # Execute the agent
            result = await self.agent.ainvoke({"messages": messages})
            
            # Extract the response
            response = result["messages"][-1].content
            
            # Update memory
            memory.add_message({"role": "assistant", "content": response})
            
            return {
                "success": True,
                "response": response,
                "agent": self.name
            }
        except Exception as e:
            error_message = format_error_for_user(e)
            
            # Update memory with the error
            memory.add_message({
                "role": "assistant", 
                "content": f"Error in {self.name}: {error_message}"
            })
            
            return {
                "success": False,
                "error": error_message,
                "agent": self.name
            }


class CoordinatorAgent:
    """Agent responsible for coordinating multiple specialized sub-agents."""
    
    def __init__(
        self, 
        model: ChatAnthropic, 
        sub_agents: Dict[str, SpecializedSubAgent],
        tool_selector: ToolSelectionAgent,
        memory: AgentMemory
    ):
        """Initialize the coordinator agent.
        
        Args:
            model: Language model to use
            sub_agents: Dictionary of sub-agents by name
            tool_selector: Tool selection agent
            memory: Agent memory
        """
        self.model = model
        self.sub_agents = sub_agents
        self.tool_selector = tool_selector
        self.memory = memory
        
        # Create the coordinator prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a coordinator agent responsible for managing multiple specialized sub-agents to complete complex tasks.
Your job is to:
1. Analyze the user's request
2. Break it down into subtasks
3. Assign each subtask to the appropriate sub-agent
4. Synthesize the results into a coherent response

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
        # Add the user request to memory
        self.memory.add_message({"role": "user", "content": request})
        
        # Get recent conversation history
        history = self.memory.get_recent_messages(5)
        
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
            result = await agent.execute(request, self.memory)
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
        
        # Add the final response to memory
        self.memory.add_message({"role": "assistant", "content": final_response})
        
        return final_response


# Factory function to create specialized sub-agents
def create_specialized_sub_agents(
    model: ChatAnthropic, 
    all_tools: List[BaseTool]
) -> Dict[str, SpecializedSubAgent]:
    """Create specialized sub-agents for different tasks.
    
    Args:
        model: Language model to use
        all_tools: List of all available tools
        
    Returns:
        Dictionary of sub-agents by name
    """
    # Categorize tools by type
    search_tools = [t for t in all_tools if any(term in t.name.lower() for term in ["search", "brave"])]
    scraping_tools = [t for t in all_tools if any(term in t.name.lower() for term in ["scrape", "extract", "web"])]
    product_tools = [t for t in all_tools if any(term in t.name.lower() for term in ["product", "amazon"])]
    social_tools = [t for t in all_tools if any(term in t.name.lower() for term in ["social", "instagram", "facebook", "twitter"])]
    
    # Create specialized sub-agents
    sub_agents = {}
    
    # Search agent
    if search_tools:
        search_prompt = """You are a specialized search agent that focuses on finding information on the web.
Your primary goal is to find accurate and relevant information in response to user queries.

When searching:
- Use the most appropriate search tool for the query
- Extract the most relevant information from search results
- Provide direct answers with sources when possible
- Be concise and focus on the most important information

Always cite your sources and provide links to where the information was found.
"""
        sub_agents["search"] = SpecializedSubAgent("Search Agent", model, search_tools, search_prompt)
    
    # Scraping agent
    if scraping_tools:
        scraping_prompt = """You are a specialized web scraping agent that focuses on extracting structured data from websites.
Your primary goal is to extract specific information from web pages in a clean, structured format.

When scraping:
- Choose the most appropriate scraping tool for the website
- Extract only the information requested by the user
- Format the data in a clean, readable way
- Handle errors gracefully and suggest alternatives when a site can't be scraped

Always respect website terms of service and be mindful of rate limits.
"""
        sub_agents["scraping"] = SpecializedSubAgent("Scraping Agent", model, scraping_tools, scraping_prompt)
    
    # Product research agent
    if product_tools:
        product_prompt = """You are a specialized product research agent that focuses on gathering and comparing product information.
Your primary goal is to help users make informed purchasing decisions by providing detailed product information.

When researching products:
- Gather comprehensive information about products
- Compare features, prices, and reviews across products
- Highlight key differences between similar products
- Provide objective assessments based on the data

Always present information in a structured, easy-to-compare format.
"""
        sub_agents["product"] = SpecializedSubAgent("Product Research Agent", model, product_tools, product_prompt)
    
    # Social media agent
    if social_tools:
        social_prompt = """You are a specialized social media analysis agent that focuses on extracting insights from social platforms.
Your primary goal is to analyze social media content and provide meaningful insights.

When analyzing social media:
- Extract relevant content from social media platforms
- Analyze engagement metrics and trends
- Identify key themes and sentiments
- Provide context for social media activity

Always respect privacy considerations and focus on public information.
"""
        sub_agents["social"] = SpecializedSubAgent("Social Media Agent", model, social_tools, social_prompt)
    
    # Default agent with all tools
    default_prompt = """You are a versatile agent with access to a wide range of tools for web automation and data collection.
Your goal is to help users with any task by using the most appropriate tools available.

When approaching tasks:
- Break down complex requests into manageable steps
- Choose the most appropriate tools for each step
- Provide clear explanations of what you're doing
- Format results in a readable, structured way

Be thorough and comprehensive in your responses.
"""
    sub_agents["default"] = SpecializedSubAgent("Default Agent", model, all_tools, default_prompt)
    
    return sub_agents