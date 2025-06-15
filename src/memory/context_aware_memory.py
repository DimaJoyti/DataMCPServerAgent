"""
Context-aware memory module for DataMCPServerAgent.
This module provides advanced memory retrieval and context management capabilities.
"""

import json
import re
import time
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.memory.memory_persistence import MemoryDatabase


class MemoryRetriever:
    """Advanced memory retrieval system with semantic search capabilities."""

    def __init__(self, model: ChatAnthropic, memory_db: MemoryDatabase):
        """Initialize the memory retriever.

        Args:
            model: Language model to use
            memory_db: Memory database for persistence
        """
        self.model = model
        self.memory_db = memory_db

        # Create the memory search prompt
        self.search_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a memory search agent responsible for finding relevant information in the agent's memory.
Your job is to analyze a user request and identify the most relevant pieces of information from the memory that would help answer it.

For each request, you should:
1. Analyze the key concepts, entities, and requirements in the request
2. Identify what types of information would be most helpful (conversation history, entity data, tool usage)
3. Determine specific search criteria for each type of information
4. Rank the relevance of different memory items

Respond with a JSON object containing:
- "conversation_keywords": Array of keywords to search for in conversation history
- "entity_types": Array of entity types that might be relevant
- "entity_ids": Array of specific entity IDs if known
- "tool_names": Array of tools that might have relevant usage history
- "time_range": Object with "start" and "end" timestamps if time-based filtering is needed
- "relevance_criteria": Description of what makes a memory item relevant to this request
"""
                ),
                HumanMessage(
                    content="""
User request: {request}

Available memory types:
- Conversation history
- Entity memory (types: {entity_types})
- Tool usage history (tools: {tool_names})

Find the most relevant information in memory for this request.
"""
                ),
            ]
        )

        # Create the memory ranking prompt
        self.ranking_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a memory ranking agent responsible for determining the relevance of memory items to a user request.
Your job is to analyze memory items and rank them by their relevance to the current request.

For each memory item, you should:
1. Analyze how directly it relates to the current request
2. Consider its recency and importance
3. Evaluate how much it contributes to answering the request
4. Assign a relevance score from 0 to 10

Respond with a JSON object containing:
- "ranked_items": Array of objects with "item_id", "relevance_score", and "reasoning"
- "top_items": Array of the most relevant item IDs
- "relevance_summary": Brief explanation of the ranking criteria used
"""
                ),
                HumanMessage(
                    content="""
User request: {request}

Memory items:
{memory_items}

Rank these memory items by relevance to the request.
"""
                ),
            ]
        )

    async def search_memory(self, request: str) -> Dict[str, Any]:
        """Search memory for information relevant to a request.

        Args:
            request: User request

        Returns:
            Dictionary of relevant memory items by type
        """
        # Get available entity types and tool names
        entity_types = list(await self.memory_db.get_entity_types())
        tool_names = list(await self.memory_db.get_tool_names())

        # Prepare the input for the search prompt
        input_values = {
            "request": request,
            "entity_types": ", ".join(entity_types),
            "tool_names": ", ".join(tool_names),
        }

        # Get search criteria from the model
        messages = self.search_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            json_str = (
                content.split("```json")[1].split("```")[0] if "```json" in content else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            search_criteria = json.loads(json_str)
        except Exception:
            # If parsing fails, use default search criteria
            search_criteria = {
                "conversation_keywords": [request.split()[:5]],  # First 5 words
                "entity_types": entity_types[:2],  # First 2 entity types
                "entity_ids": [],
                "tool_names": tool_names[:3],  # First 3 tools
                "time_range": {
                    "start": time.time() - 86400,
                    "end": time.time(),
                },  # Last 24 hours
                "relevance_criteria": "Direct relevance to the request",
            }

        # Retrieve relevant memory items
        relevant_items = {
            "conversation": await self._search_conversation(request, search_criteria),
            "entities": await self._search_entities(request, search_criteria),
            "tool_usage": await self._search_tool_usage(request, search_criteria),
        }

        # Rank the memory items
        ranked_items = await self._rank_memory_items(request, relevant_items)

        return {
            "relevant_items": relevant_items,
            "ranked_items": ranked_items,
            "search_criteria": search_criteria,
        }

    async def _search_conversation(
        self, request: str, search_criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search conversation history for relevant messages.

        Args:
            request: User request
            search_criteria: Search criteria

        Returns:
            List of relevant conversation messages
        """
        # Get conversation history
        conversation = await self.memory_db.load_conversation_history()

        # Filter by keywords
        keywords = search_criteria.get("conversation_keywords", [])
        relevant_messages = []

        for message in conversation:
            # Check if any keyword is in the message
            if any(keyword.lower() in message["content"].lower() for keyword in keywords):
                relevant_messages.append(message)

        # If no messages match keywords, return the most recent messages
        if not relevant_messages and conversation:
            relevant_messages = conversation[-5:]

        return relevant_messages

    async def _search_entities(
        self, request: str, search_criteria: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Search entity memory for relevant entities.

        Args:
            request: User request
            search_criteria: Search criteria

        Returns:
            Dictionary of relevant entities by ID
        """
        # Get entity types to search
        entity_types = search_criteria.get("entity_types", [])
        entity_ids = search_criteria.get("entity_ids", [])

        relevant_entities = {}

        # If specific entity IDs are provided, retrieve those entities
        if entity_ids:
            for entity_id in entity_ids:
                # Try to determine the entity type from the ID format
                for entity_type in entity_types:
                    entity = await self.memory_db.load_entity(entity_type, entity_id)
                    if entity:
                        relevant_entities[entity_id] = entity
                        break

        # Otherwise, retrieve entities by type
        else:
            for entity_type in entity_types:
                entities = await self.memory_db.load_entities_by_type(entity_type)
                relevant_entities.update(entities)

        return relevant_entities

    async def _search_tool_usage(
        self, request: str, search_criteria: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search tool usage history for relevant tool executions.

        Args:
            request: User request
            search_criteria: Search criteria

        Returns:
            Dictionary of relevant tool usage by tool name
        """
        # Get tool names to search
        tool_names = search_criteria.get("tool_names", [])
        time_range = search_criteria.get("time_range", {"start": 0, "end": time.time()})

        relevant_tool_usage = {}

        # Retrieve tool usage for each tool
        for tool_name in tool_names:
            tool_usage = await self.memory_db.load_tool_usage(tool_name)

            # Filter by time range
            if tool_name in tool_usage:
                filtered_usage = [
                    usage
                    for usage in tool_usage[tool_name]
                    if time_range["start"] <= usage["timestamp"] <= time_range["end"]
                ]

                if filtered_usage:
                    relevant_tool_usage[tool_name] = filtered_usage

        return relevant_tool_usage

    async def _rank_memory_items(
        self, request: str, memory_items: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank memory items by relevance to the request.

        Args:
            request: User request
            memory_items: Dictionary of memory items by type

        Returns:
            List of ranked memory items
        """
        # Format memory items for the ranking prompt
        formatted_items = self._format_memory_items(memory_items)

        # If there are no memory items, return an empty list
        if not formatted_items:
            return []

        # Prepare the input for the ranking prompt
        input_values = {"request": request, "memory_items": formatted_items}

        # Get ranking from the model
        messages = self.ranking_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            json_str = (
                content.split("```json")[1].split("```")[0] if "```json" in content else content
            )
            json_str = json_str.strip()

            # Handle cases where the JSON might be embedded in text
            if not json_str.startswith("{"):
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]

            ranking = json.loads(json_str)
            return ranking.get("ranked_items", [])
        except Exception:
            # If parsing fails, return a simple ranking
            return [
                {
                    "item_id": f"item_{i}",
                    "relevance_score": 5,
                    "reasoning": "Default ranking",
                }
                for i in range(len(formatted_items.split("\n\n")))
            ]

    def _format_memory_items(self, memory_items: Dict[str, Any]) -> str:
        """Format memory items for the ranking prompt.

        Args:
            memory_items: Dictionary of memory items by type

        Returns:
            Formatted memory items
        """
        formatted = ""

        # Format conversation messages
        if "conversation" in memory_items and memory_items["conversation"]:
            formatted += "## Conversation History\n\n"
            for i, message in enumerate(memory_items["conversation"], 1):
                formatted += f"### Message {i} (item_conv_{i})\n"
                formatted += f"Role: {message['role']}\n"
                formatted += f"Content: {message['content'][:100]}...\n\n"

        # Format entities
        if "entities" in memory_items and memory_items["entities"]:
            formatted += "## Entity Memory\n\n"
            for i, (entity_id, entity) in enumerate(memory_items["entities"].items(), 1):
                formatted += f"### Entity {i} (item_entity_{i})\n"
                formatted += f"ID: {entity_id}\n"
                formatted += f"Data: {json.dumps(entity)[:100]}...\n\n"

        # Format tool usage
        if "tool_usage" in memory_items and memory_items["tool_usage"]:
            formatted += "## Tool Usage History\n\n"
            for i, (tool_name, usages) in enumerate(memory_items["tool_usage"].items(), 1):
                for j, usage in enumerate(usages[:3], 1):  # Limit to 3 usages per tool
                    formatted += f"### Tool Usage {i}.{j} (item_tool_{i}_{j})\n"
                    formatted += f"Tool: {tool_name}\n"
                    formatted += f"Args: {json.dumps(usage['args'])[:50]}...\n"
                    formatted += f"Result: {str(usage['result'])[:50]}...\n\n"

        return formatted

    def get_memory_types(self) -> List[str]:
        """Get the types of memory available.

        Returns:
            List of memory types
        """
        return ["conversation", "entities", "tool_usage"]


class ContextManager:
    """Manager for maintaining and updating context during agent execution."""

    def __init__(self, memory_retriever: MemoryRetriever):
        """Initialize the context manager.

        Args:
            memory_retriever: Memory retriever
        """
        self.memory_retriever = memory_retriever
        self.current_context = {
            "conversation": [],
            "entities": {},
            "tool_usage": {},
            "working_memory": {},
        }

    async def update_context(self, request: str) -> Dict[str, Any]:
        """Update the current context based on a new request.

        Args:
            request: User request

        Returns:
            Updated context
        """
        # Search memory for relevant information
        memory_search = await self.memory_retriever.search_memory(request)

        # Update conversation context
        if "conversation" in memory_search["relevant_items"]:
            self.current_context["conversation"] = memory_search["relevant_items"]["conversation"]

        # Update entity context
        if "entities" in memory_search["relevant_items"]:
            self.current_context["entities"].update(memory_search["relevant_items"]["entities"])

        # Update tool usage context
        if "tool_usage" in memory_search["relevant_items"]:
            self.current_context["tool_usage"].update(memory_search["relevant_items"]["tool_usage"])

        # Extract entities from the request and add them to working memory
        entities = self._extract_entities(request)
        self.current_context["working_memory"].update(entities)

        return self.current_context

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Extract dates
        date_pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
        dates = re.findall(date_pattern, text)
        if dates:
            entities["dates"] = dates

        # Extract numbers
        number_pattern = r"\b(\d+(?:\.\d+)?)\b"
        numbers = re.findall(number_pattern, text)
        if numbers:
            entities["numbers"] = [float(n) for n in numbers]

        # Extract URLs
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)
        if urls:
            entities["urls"] = urls

        # Extract potential product names (capitalized phrases)
        product_pattern = r"\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        products = re.findall(product_pattern, text)
        if products:
            entities["products"] = products

        return entities

    def get_formatted_context(self) -> str:
        """Get a formatted representation of the current context.

        Returns:
            Formatted context
        """
        formatted = "# Current Context\n\n"

        # Format conversation context
        if self.current_context["conversation"]:
            formatted += "## Recent Conversation\n\n"
            for message in self.current_context["conversation"][-3:]:  # Last 3 messages
                formatted += f"**{message['role']}**: {message['content'][:100]}...\n\n"

        # Format entity context
        if self.current_context["entities"]:
            formatted += "## Relevant Entities\n\n"
            for entity_id, entity in list(self.current_context["entities"].items())[
                :5
            ]:  # First 5 entities
                formatted += f"- **{entity_id}**: {json.dumps(entity)[:100]}...\n"
            formatted += "\n"

        # Format tool usage context
        if self.current_context["tool_usage"]:
            formatted += "## Recent Tool Usage\n\n"
            for tool_name, usages in self.current_context["tool_usage"].items():
                formatted += f"### {tool_name}\n"
                for usage in usages[:2]:  # First 2 usages per tool
                    formatted += f"- Args: {json.dumps(usage['args'])[:50]}...\n"
                    formatted += f"- Result: {str(usage['result'])[:50]}...\n"
                formatted += "\n"

        # Format working memory
        if self.current_context["working_memory"]:
            formatted += "## Working Memory\n\n"
            for key, value in self.current_context["working_memory"].items():
                formatted += f"- **{key}**: {value}\n"

        return formatted

    def add_to_working_memory(self, key: str, value: Any) -> None:
        """Add an item to working memory.

        Args:
            key: Key for the item
            value: Value to store
        """
        self.current_context["working_memory"][key] = value

    def clear_working_memory(self) -> None:
        """Clear the working memory."""
        self.current_context["working_memory"] = {}
