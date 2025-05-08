"""
Enhanced tool selection module for DataMCPServerAgent.
This module provides advanced tool selection algorithms with historical performance tracking.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from src.memory.memory_persistence import MemoryDatabase


class ToolPerformanceTracker:
    """Tracker for tool performance metrics."""
    
    def __init__(self, db: MemoryDatabase):
        """Initialize the tool performance tracker.
        
        Args:
            db: Memory database for persistence
        """
        self.db = db
        self.current_executions = {}
    
    def start_execution(self, tool_name: str) -> None:
        """Start tracking execution time for a tool.
        
        Args:
            tool_name: Name of the tool
        """
        self.current_executions[tool_name] = time.time()
    
    def end_execution(self, tool_name: str, success: bool) -> float:
        """End tracking execution time for a tool and save performance metrics.
        
        Args:
            tool_name: Name of the tool
            success: Whether the execution was successful
            
        Returns:
            Execution time in seconds
        """
        if tool_name not in self.current_executions:
            # If we don't have a start time, use a default execution time
            execution_time = 1.0
        else:
            execution_time = time.time() - self.current_executions[tool_name]
            del self.current_executions[tool_name]
        
        # Save performance metrics to the database
        self.db.save_tool_performance(tool_name, success, execution_time)
        
        return execution_time
    
    def get_performance(self, tool_name: str) -> Dict[str, Any]:
        """Get performance metrics for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Performance metrics
        """
        return self.db.get_tool_performance(tool_name)


class EnhancedToolSelector:
    """Enhanced tool selector with learning capabilities."""
    
    def __init__(
        self, 
        model: ChatAnthropic, 
        tools: List[BaseTool],
        db: MemoryDatabase,
        performance_tracker: ToolPerformanceTracker
    ):
        """Initialize the enhanced tool selector.
        
        Args:
            model: Language model to use
            tools: List of available tools
            db: Memory database for persistence
            performance_tracker: Tool performance tracker
        """
        self.model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.db = db
        self.performance_tracker = performance_tracker
        
        # Create the tool selection prompt
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an advanced tool selection agent responsible for choosing the most appropriate tools for a given task.
Your job is to analyze the user's request and determine which tools would be most effective for completing it.

For each request, you should:
1. Analyze the task requirements in detail
2. Consider the available tools and their capabilities
3. Review the historical performance of each tool
4. Select 1-3 tools that would be most appropriate for the task
5. Explain your reasoning for each tool selection

Respond with a JSON object containing:
- "selected_tools": Array of tool names
- "reasoning": Detailed explanation of your selection
- "execution_order": Suggested order to use the tools (array of tool names)
- "fallback_tools": Array of alternative tools to try if the primary tools fail

Be strategic in your selection - choose tools that:
- Have high success rates for similar tasks
- Complement each other and cover all aspects of the task
- Have reasonable execution times
- Are specialized for the specific domain of the task
"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="""
User request: {request}

Available tools:
{tool_descriptions}

Tool performance metrics:
{tool_performance}

Select the most appropriate tools for this task.
""")
        ])
        
        # Create the learning feedback prompt
        self.feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a tool selection improvement agent responsible for learning from past tool usage.
Your job is to analyze the results of tool executions and provide feedback to improve future tool selection.

For each tool execution, you should:
1. Analyze whether the tool was appropriate for the task
2. Identify any issues or inefficiencies in the tool usage
3. Suggest improvements for future tool selection
4. Provide a confidence score for your feedback (0-100)

Respond with a JSON object containing:
- "appropriate": Boolean indicating whether the tool was appropriate for the task
- "issues": Array of identified issues
- "suggestions": Array of suggestions for improvement
- "confidence": Confidence score (0-100)
- "learning_points": Key learning points from this execution
"""),
            HumanMessage(content="""
Original request: {request}
Selected tool: {tool_name}
Tool arguments: {tool_args}
Tool result: {tool_result}
Execution time: {execution_time} seconds
Success: {success}

Provide feedback on this tool execution.
""")
        ])
    
    async def select_tools(
        self, 
        request: str, 
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
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
        
        # Get performance metrics for each tool
        tool_performance = "\n\n".join([
            self._format_tool_performance(tool.name) for tool in self.tools
        ])
        
        # Prepare the input for the prompt
        input_values = {
            "request": request,
            "tool_descriptions": tool_descriptions,
            "tool_performance": tool_performance,
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
            if "fallback_tools" not in result:
                result["fallback_tools"] = []
            
            # Filter out any tools that don't exist
            result["selected_tools"] = [t for t in result["selected_tools"] if t in self.tool_map]
            result["execution_order"] = [t for t in result["execution_order"] if t in self.tool_map]
            result["fallback_tools"] = [t for t in result["fallback_tools"] if t in self.tool_map]
            
            return result
        except Exception as e:
            # If parsing fails, return a default selection with fallbacks
            return self._get_default_selection(str(e))
    
    def _get_default_selection(self, error_message: str) -> Dict[str, Any]:
        """Get a default tool selection when parsing fails.
        
        Args:
            error_message: Error message from parsing
            
        Returns:
            Default tool selection
        """
        # Get the top 3 tools by success rate
        top_tools = self._get_top_tools_by_success_rate(3)
        
        if not top_tools:
            # If we don't have performance data, use the first available tool
            selected_tools = [self.tools[0].name] if self.tools else []
            fallback_tools = [self.tools[1].name] if len(self.tools) > 1 else []
        else:
            selected_tools = [top_tools[0]]
            fallback_tools = top_tools[1:] if len(top_tools) > 1 else []
        
        return {
            "selected_tools": selected_tools,
            "reasoning": f"Error parsing tool selection: {error_message}. Using tools with highest success rates.",
            "execution_order": selected_tools,
            "fallback_tools": fallback_tools
        }
    
    def _get_top_tools_by_success_rate(self, n: int = 3) -> List[str]:
        """Get the top N tools by success rate.
        
        Args:
            n: Number of tools to return
            
        Returns:
            List of tool names
        """
        tool_metrics = []
        
        for tool in self.tools:
            metrics = self.performance_tracker.get_performance(tool.name)
            
            # Only consider tools that have been used at least once
            if metrics["total_uses"] > 0:
                tool_metrics.append((tool.name, metrics["success_rate"]))
        
        # Sort by success rate (descending)
        tool_metrics.sort(key=lambda x: x[1], reverse=True)
        
        return [tool for tool, _ in tool_metrics[:n]]
    
    def _format_tool_performance(self, tool_name: str) -> str:
        """Format tool performance metrics for inclusion in the prompt.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Formatted performance metrics
        """
        metrics = self.performance_tracker.get_performance(tool_name)
        
        if metrics["total_uses"] == 0:
            return f"{tool_name}: No usage data available"
        
        return f"{tool_name}:\n" \
               f"  - Success rate: {metrics['success_rate']:.2f}%\n" \
               f"  - Total uses: {metrics['total_uses']}\n" \
               f"  - Avg execution time: {metrics['avg_execution_time']:.2f}s"
    
    async def provide_execution_feedback(
        self,
        request: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        execution_time: float,
        success: bool
    ) -> Dict[str, Any]:
        """Provide feedback on a tool execution to improve future selection.
        
        Args:
            request: Original user request
            tool_name: Name of the tool used
            tool_args: Arguments passed to the tool
            tool_result: Result returned by the tool
            execution_time: Time taken to execute the tool
            success: Whether the execution was successful
            
        Returns:
            Feedback data
        """
        # Prepare the input for the feedback prompt
        input_values = {
            "request": request,
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args),
            "tool_result": str(tool_result)[:500],  # Limit result size
            "execution_time": execution_time,
            "success": success
        }
        
        # Get the feedback from the model
        messages = self.feedback_prompt.format_messages(**input_values)
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
            
            feedback = json.loads(json_str)
            
            # Save the feedback to the database
            self.db.save_learning_feedback(
                "tool_selector",
                "execution_feedback",
                {
                    "request": request,
                    "tool_name": tool_name,
                    "success": success,
                    "feedback": feedback
                }
            )
            
            return feedback
        except Exception as e:
            # If parsing fails, return a default feedback
            default_feedback = {
                "appropriate": success,
                "issues": ["Error parsing feedback"],
                "suggestions": ["Improve feedback parsing"],
                "confidence": 50,
                "learning_points": [f"Error in feedback generation: {str(e)}"]
            }
            
            # Save the default feedback to the database
            self.db.save_learning_feedback(
                "tool_selector",
                "execution_feedback",
                {
                    "request": request,
                    "tool_name": tool_name,
                    "success": success,
                    "feedback": default_feedback,
                    "error": str(e)
                }
            )
            
            return default_feedback