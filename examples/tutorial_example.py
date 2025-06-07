"""
Comprehensive tutorial example for DataMCPServerAgent.

This example demonstrates how to:
1. Set up and configure the agent
2. Create and register custom tools
3. Use the enhanced tool selection system
4. Integrate with the memory system
5. Provide feedback for learning
6. Extend the agent with new capabilities

Each section includes detailed comments explaining the concepts and implementation.
"""

import asyncio
import os
import sys
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.memory.memory_persistence import MemoryDatabase
from src.tools.enhanced_tool_selection import EnhancedToolSelector, ToolPerformanceTracker

# =====================================================================
# SECTION 1: Creating Custom Tools
# =====================================================================

class NoteTakingTool(BaseTool):
    """Tool for taking and retrieving notes."""

    name = "note_taking"
    description = "Take notes and retrieve them later"

    def __init__(self):
        """Initialize the note taking tool."""
        self.notes = {}
        super().__init__()

    async def _arun(self, action: str, title: Optional[str] = None, content: Optional[str] = None) -> str:
        """Run the note taking tool asynchronously.

        Args:
            action: Action to perform (save, retrieve, list, delete)
            title: Title of the note (for save, retrieve, delete)
            content: Content of the note (for save)

        Returns:
            Result of the action
        """
        if action == "save":
            if not title or not content:
                return "Error: Title and content are required for saving a note"

            self.notes[title] = content
            return f"Note '{title}' saved successfully"

        elif action == "retrieve":
            if not title:
                return "Error: Title is required for retrieving a note"

            if title not in self.notes:
                return f"Error: Note '{title}' not found"

            return f"## Note: {title}\n\n{self.notes[title]}"

        elif action == "list":
            if not self.notes:
                return "No notes found"

            result = "## Available Notes\n\n"
            for note_title in self.notes.keys():
                result += f"- {note_title}\n"

            return result

        elif action == "delete":
            if not title:
                return "Error: Title is required for deleting a note"

            if title not in self.notes:
                return f"Error: Note '{title}' not found"

            del self.notes[title]
            return f"Note '{title}' deleted successfully"

        else:
            return f"Error: Unknown action '{action}'. Valid actions are: save, retrieve, list, delete"

class TaskManagerTool(BaseTool):
    """Tool for managing tasks."""

    name = "task_manager"
    description = "Manage tasks (add, complete, list, delete)"

    def __init__(self):
        """Initialize the task manager tool."""
        self.tasks = {}  # Dictionary of tasks with status
        self.task_id_counter = 1
        super().__init__()

    async def _arun(self, action: str, task: Optional[str] = None, task_id: Optional[int] = None) -> str:
        """Run the task manager tool asynchronously.

        Args:
            action: Action to perform (add, complete, list, delete)
            task: Task description (for add)
            task_id: Task ID (for complete, delete)

        Returns:
            Result of the action
        """
        if action == "add":
            if not task:
                return "Error: Task description is required for adding a task"

            task_id = self.task_id_counter
            self.tasks[task_id] = {"description": task, "completed": False}
            self.task_id_counter += 1

            return f"Task added with ID {task_id}: {task}"

        elif action == "complete":
            if not task_id:
                return "Error: Task ID is required for completing a task"

            if task_id not in self.tasks:
                return f"Error: Task with ID {task_id} not found"

            self.tasks[task_id]["completed"] = True
            return f"Task {task_id} marked as completed"

        elif action == "list":
            if not self.tasks:
                return "No tasks found"

            result = "## Tasks\n\n"
            for tid, task_info in self.tasks.items():
                status = "✅" if task_info["completed"] else "⬜"
                result += f"{status} {tid}: {task_info['description']}\n"

            return result

        elif action == "delete":
            if not task_id:
                return "Error: Task ID is required for deleting a task"

            if task_id not in self.tasks:
                return f"Error: Task with ID {task_id} not found"

            del self.tasks[task_id]
            return f"Task {task_id} deleted successfully"

        else:
            return f"Error: Unknown action '{action}'. Valid actions are: add, complete, list, delete"

# =====================================================================
# SECTION 2: Tool Provider
# =====================================================================

class ProductivityToolProvider:
    """Provider for productivity tools."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the tool provider.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.note_taking_tool = NoteTakingTool()
        self.task_manager_tool = TaskManagerTool()

    async def get_tools(self) -> List[BaseTool]:
        """Get the tools provided by this provider.

        Returns:
            List of tools
        """
        tools = []

        # Add note taking tool if enabled
        if self.config.get("enable_note_taking", True):
            tools.append(self.note_taking_tool)

        # Add task manager tool if enabled
        if self.config.get("enable_task_manager", True):
            tools.append(self.task_manager_tool)

        return tools

# =====================================================================
# SECTION 3: Custom Memory Integration
# =====================================================================

class UserPreferenceTracker:
    """Tracker for user preferences."""

    def __init__(self, db: MemoryDatabase):
        """Initialize the user preference tracker.

        Args:
            db: Memory database for persistence
        """
        self.db = db
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> Dict[str, Any]:
        """Load preferences from the database.

        Returns:
            Dictionary of preferences
        """
        preferences = self.db.get_memory("user_preferences")
        if not preferences:
            # Default preferences
            preferences = {
                "response_style": "balanced",
                "verbosity": "medium",
                "code_examples": True,
                "favorite_tools": []
            }
            self.db.save_memory("user_preferences", preferences)

        return preferences

    def get_preference(self, key: str) -> Any:
        """Get a user preference.

        Args:
            key: Preference key

        Returns:
            Preference value
        """
        return self.preferences.get(key)

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference.

        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self.db.save_memory("user_preferences", self.preferences)

    def track_tool_usage(self, tool_name: str) -> None:
        """Track tool usage to update favorite tools.

        Args:
            tool_name: Name of the tool used
        """
        favorite_tools = self.preferences.get("favorite_tools", [])

        # Update favorite tools based on usage
        if tool_name in favorite_tools:
            # Move to the front of the list
            favorite_tools.remove(tool_name)
            favorite_tools.insert(0, tool_name)
        else:
            # Add to the front of the list
            favorite_tools.insert(0, tool_name)

            # Keep only the top 5 favorite tools
            if len(favorite_tools) > 5:
                favorite_tools = favorite_tools[:5]

        self.preferences["favorite_tools"] = favorite_tools
        self.db.save_memory("user_preferences", self.preferences)

# =====================================================================
# SECTION 4: Custom Conversation Handler
# =====================================================================

class ConversationHandler:
    """Handler for conversations with the agent."""

    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        db: MemoryDatabase,
        performance_tracker: ToolPerformanceTracker,
        preference_tracker: UserPreferenceTracker
    ):
        """Initialize the conversation handler.

        Args:
            model: Language model
            tools: List of available tools
            db: Memory database
            performance_tracker: Tool performance tracker
            preference_tracker: User preference tracker
        """
        self.model = model
        self.tools = tools
        self.db = db
        self.performance_tracker = performance_tracker
        self.preference_tracker = preference_tracker
        self.tool_selector = EnhancedToolSelector(
            model=model,
            tools=tools,
            db=db,
            performance_tracker=performance_tracker
        )
        self.conversation_history = []

    async def process_message(self, message: str) -> str:
        """Process a user message.

        Args:
            message: User message

        Returns:
            Agent response
        """
        # Add the user message to the conversation history
        self.conversation_history.append(HumanMessage(content=message))

        # Check for special commands
        if message.lower() in ["exit", "quit"]:
            return "Goodbye! Thank you for using the DataMCPServerAgent."

        if message.lower() == "help":
            return self._get_help_message()

        if message.lower() == "preferences":
            return self._get_preferences()

        if message.lower().startswith("preferences set "):
            return self._set_preference(message)

        # Select tools for the message
        tool_selection = await self.tool_selector.select_tools(
            request=message,
            history=self.conversation_history
        )

        # If tools were selected, use them
        if tool_selection["selected_tools"]:
            response = await self._use_tools(message, tool_selection)
        else:
            # Otherwise, use the model directly
            response = await self._get_model_response(message)

        # Add the agent response to the conversation history
        self.conversation_history.append(AIMessage(content=response))

        # Save the conversation to the database
        self.db.save_memory(
            "conversations",
            {
                "user_message": message,
                "agent_response": response,
                "timestamp": "2023-01-01T00:00:00Z"  # Use actual timestamp in real implementation
            }
        )

        return response

    async def _use_tools(self, message: str, tool_selection: Dict[str, Any]) -> str:
        """Use selected tools to process the message.

        Args:
            message: User message
            tool_selection: Tool selection result

        Returns:
            Response after using tools
        """
        responses = []

        for tool_name in tool_selection["execution_order"]:
            tool = next((t for t in self.tools if t.name == tool_name), None)

            if not tool:
                continue

            # Start tracking execution
            self.performance_tracker.start_execution(tool_name)

            try:
                # For simplicity, we're using a basic approach to extract arguments
                # In a real implementation, you would use a more sophisticated approach

                if tool_name == "note_taking":
                    if "save" in message.lower():
                        # Extract title and content
                        title = "Example Note"
                        content = "This is an example note content."
                        result = await tool.ainvoke({"action": "save", "title": title, "content": content})
                    elif "retrieve" in message.lower():
                        # Extract title
                        title = "Example Note"
                        result = await tool.ainvoke({"action": "retrieve", "title": title})
                    elif "list" in message.lower():
                        result = await tool.ainvoke({"action": "list"})
                    elif "delete" in message.lower():
                        # Extract title
                        title = "Example Note"
                        result = await tool.ainvoke({"action": "delete", "title": title})
                    else:
                        result = await tool.ainvoke({"action": "list"})

                elif tool_name == "task_manager":
                    if "add" in message.lower():
                        # Extract task
                        task = "Example task"
                        result = await tool.ainvoke({"action": "add", "task": task})
                    elif "complete" in message.lower():
                        # Extract task ID
                        task_id = 1
                        result = await tool.ainvoke({"action": "complete", "task_id": task_id})
                    elif "list" in message.lower():
                        result = await tool.ainvoke({"action": "list"})
                    elif "delete" in message.lower():
                        # Extract task ID
                        task_id = 1
                        result = await tool.ainvoke({"action": "delete", "task_id": task_id})
                    else:
                        result = await tool.ainvoke({"action": "list"})

                else:
                    # Generic invocation for other tools
                    result = await tool.ainvoke({"query": message})

                success = True
            except Exception as e:
                result = f"Error using {tool_name}: {str(e)}"
                success = False

            # End tracking execution
            execution_time = self.performance_tracker.end_execution(tool_name, success)

            # Track tool usage for user preferences
            self.preference_tracker.track_tool_usage(tool_name)

            # Get feedback on the execution
            await self.tool_selector.provide_execution_feedback(
                request=message,
                tool_name=tool_name,
                tool_args={},  # Simplified for the example
                tool_result=result,
                execution_time=execution_time,
                success=success
            )

            responses.append(result)

        # Combine the responses
        combined_response = "\n\n".join(responses)

        # If no tools were successfully used, fall back to the model
        if not combined_response:
            return await self._get_model_response(message)

        return combined_response

    async def _get_model_response(self, message: str) -> str:
        """Get a response from the model.

        Args:
            message: User message

        Returns:
            Model response
        """
        # Get user preferences
        response_style = self.preference_tracker.get_preference("response_style")
        verbosity = self.preference_tracker.get_preference("verbosity")

        # Adjust the system message based on preferences
        system_message = f"""You are a helpful assistant.

        Response style: {response_style}
        Verbosity: {verbosity}

        Please respond to the user's message accordingly.
        """

        # Get the response from the model
        response = await self.model.ainvoke([
            {"role": "system", "content": system_message},
            *[{"role": m.type, "content": m.content} for m in self.conversation_history],
            {"role": "user", "content": message}
        ])

        return response.content

    def _get_help_message(self) -> str:
        """Get the help message.

        Returns:
            Help message
        """
        return """
        # Available Commands

        - `exit` or `quit`: End the chat session
        - `help`: Display this help message
        - `preferences`: View your preferences
        - `preferences set <key> <value>`: Set a preference

        # Available Tools

        """ + "\n".join([f"- `{tool.name}`: {tool.description}" for tool in self.tools])

    def _get_preferences(self) -> str:
        """Get the user preferences.

        Returns:
            User preferences
        """
        preferences = self.preference_tracker.preferences

        result = "## Your Preferences\n\n"

        for key, value in preferences.items():
            result += f"- **{key}**: {value}\n"

        return result

    def _set_preference(self, message: str) -> str:
        """Set a user preference.

        Args:
            message: User message

        Returns:
            Confirmation message
        """
        # Extract key and value
        parts = message.split(" ", 3)

        if len(parts) < 4:
            return "Error: Invalid format. Use 'preferences set <key> <value>'."

        key = parts[2]
        value = parts[3]

        # Convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)

        # Set the preference
        self.preference_tracker.set_preference(key, value)

        return f"Preference '{key}' set to '{value}'."

# =====================================================================
# SECTION 5: Main Tutorial Example
# =====================================================================

async def run_tutorial():
    """Run the tutorial example."""
    print("Running tutorial example...")

    # Step 1: Initialize dependencies
    model = ChatAnthropic(model="claude-3-sonnet-20240229")
    db = MemoryDatabase()
    performance_tracker = ToolPerformanceTracker(db)

    # Step 2: Create custom tools
    note_taking_tool = NoteTakingTool()
    task_manager_tool = TaskManagerTool()

    # Step 3: Create the tool provider
    provider_config = {
        "enable_note_taking": True,
        "enable_task_manager": True
    }
    tool_provider = ProductivityToolProvider(provider_config)

    # Step 4: Get tools from the provider
    tools = await tool_provider.get_tools()

    # Step 5: Initialize the user preference tracker
    preference_tracker = UserPreferenceTracker(db)

    # Step 6: Initialize the conversation handler
    conversation_handler = ConversationHandler(
        model=model,
        tools=tools,
        db=db,
        performance_tracker=performance_tracker,
        preference_tracker=preference_tracker
    )

    # Step 7: Process some example messages
    example_messages = [
        "Hello! What can you help me with?",
        "Can you take a note for me?",
        "Add a task to buy groceries",
        "Show me my tasks",
        "What are my preferences?",
        "preferences set response_style concise",
        "help"
    ]

    for message in example_messages:
        print(f"\nUser: {message}")
        response = await conversation_handler.process_message(message)
        print(f"Agent: {response}")

    print("\nTutorial completed!")

async def run_agent_with_tutorial_features():
    """Run the agent with the tutorial features."""
    print("Running agent with tutorial features...")

    # Initialize dependencies
    db = MemoryDatabase()
    performance_tracker = ToolPerformanceTracker(db)

    # Create custom tools
    note_taking_tool = NoteTakingTool()
    task_manager_tool = TaskManagerTool()

    # Configure the agent
    config = {
        "initial_prompt": """
        You are an assistant with productivity tools.
        You can use the note_taking tool to take and retrieve notes.
        You can use the task_manager tool to manage tasks.

        Try to be helpful and use the appropriate tools when needed.
        """,
        "additional_tools": [note_taking_tool, task_manager_tool],
        "memory_db": db,
        "performance_tracker": performance_tracker,
        "tool_selection_strategy": "enhanced",
        "verbose": True
    }

    # Run the agent
    await chat_with_advanced_enhanced_agent(config=config)

if __name__ == "__main__":
    # Choose which example to run
    example_type = "tutorial"  # Change to "agent" to run the agent example

    if example_type == "tutorial":
        asyncio.run(run_tutorial())
    elif example_type == "agent":
        asyncio.run(run_agent_with_tutorial_features())
    else:
        print(f"Unknown example type: {example_type}")
